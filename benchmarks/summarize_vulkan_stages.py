"""Summarize Vulkan API time within measured NVTX stages from an Nsight SQLite export.

These are host API intervals, not GPU kernel durations. Only measured_step
ranges are included; process startup and warmup are excluded.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('database',type=Path)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
c=sqlite3.connect(str(a.database))
ranges=c.execute('select n.start,n.end,coalesce(n.text,s.value) from NVTX_EVENTS n left join StringIds s on n.textId=s.id where n.end is not null').fetchall()
steps=[(start,end) for start,end,name in ranges if name=='measured_step']
assert steps,'No measured_step NVTX ranges found'
report={}
for stage in ['zero_grad','forward','loss','backward','optimizer']:
 selected=[(start,end) for start,end,name in ranges if name==stage and any(start>=s and end<=e for s,e in steps)]
 assert len(selected)==len(steps),(stage,len(selected),len(steps))
 calls=defaultdict(lambda:dict(calls=0,ms=0))
 for start,end in selected:
  for name,n,ns in c.execute('select s.value,count(*),sum(a.end-a.start) from VULKAN_API a join StringIds s on a.nameId=s.id where a.start>=? and a.end<=? group by s.value',(start,end)):
   calls[name]['calls']+=n/len(steps);calls[name]['ms']+=ns/1e6/len(steps)
 report[stage]=dict(mean_wall_ms=sum(e-s for s,e in selected)/1e6/len(steps),
                    api_per_step=dict(sorted(calls.items(),key=lambda v:v[1]['ms'],reverse=True)))
result=dict(measured_steps=len(steps),measurement='Host Vulkan API time; includes waits; not GPU kernel time',stages=report)
a.out.parent.mkdir(parents=True,exist_ok=True)
a.out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({stage:dict(wall=v['mean_wall_ms'],api=list(v['api_per_step'].items())[:5]) for stage,v in report.items()},indent=2))

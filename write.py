#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from collections import defaultdict


def write_inference_html(args):
    with open(f'{args.experiment_path}/test_index.html', 'w') as f:
        f.write('<html>')
        with open('web/header.html', 'r') as header:
            f.write(header.read())
        f.write('<body><table class="table" style="width: 100%; white-space: nowrap;"><thead><tr style="background-color:orange;"><th>frame <input type="checkbox" onchange="hideColumn(1);"></th>')

        now_files = [(x.split('_')[0], x) for x in os.listdir(f'{args.experiment_path}/test_outputs/')]
        files_dict = defaultdict(lambda: defaultdict(list))
        for iii, name in now_files:
            files_dict[iii]['now'].append(name)

        indices = sorted(list(set([x[0] for x in now_files])), key=lambda x: int(x.split('_')[0]))
        for qqq, iii in enumerate(indices):
            now_files = sorted([x for x in files_dict[iii]['now']])
            if qqq == 0:
                my_len = len(now_files)
                for z, x in enumerate(now_files):
                    f.write(f'<th>{x.split("_")[-1].split(".")[0]} <input type="checkbox" onchange="hideColumn({z+2});"></th>')
                f.write(f'</tr></thead><tbody>')

            f.write(f'<tr><td><div><span style="width: 2vw;">{str(qqq)}</span></div></td>')
            for z, x in enumerate(now_files):
                f.write(f'<td><div><img class="lazy" onerror="this.onerror=null; this.remove();" data-src="/~relh/experiments/{args.name}/test_outputs/{x}"></div></td>')
            f.write('</tr>')

        f.write('</tbody></table>')
        f.write('''<script>
                var lazyloadinstance = new LazyLoad({
                  // Your custom settings go here
                });
        </script>''')
        f.write('</div></body></html>')


def write_index_html(args):
    with open(f'{args.experiment_path}/index.html', 'w') as f:
        f.write('<html>')
        with open('web/header.html', 'r') as header:
            f.write(header.read())
        f.write('<body><table class="table" style="width: 100%; white-space: nowrap;"><thead><tr style="background-color:chartreuse;"><th>frame <input type="checkbox" onchange="hideColumn(1);"></th>')

        now_files = [(x.split('_')[0], x) for x in os.listdir(f'{args.experiment_path}outputs/') if 'now' in x]
        future_files = [(x.split('_')[0], x) for x in os.listdir(f'{args.experiment_path}outputs/') if 'future' in x]
        files_dict = defaultdict(lambda: defaultdict(list))
        for iii, name in now_files:
            files_dict[iii]['now'].append(name)
        for iii, name in future_files:
            files_dict[iii]['future'].append(name)

        indices = sorted(list(set([x[0] for x in now_files])), key=lambda x: int(x.split('_')[0]))
        for qqq, iii in enumerate(indices):
            #print(iii)
            for b in range(args.batch_size // args.num_gpus):
                now_files = sorted([x for x in files_dict[iii]['now'] if int(x.split('_')[1]) == b])
                if iii == '0' and b == 0:
                    continue
                if iii == '0' and b == 1:
                    my_len = len(now_files)
                    for z, x in enumerate(now_files):
                        f.write(f'<th>{x.split("_")[-1].split(".")[0]} <input type="checkbox" onchange="hideColumn({z+2});"></th>')
                    f.write(f'</tr></thead><tbody>')

                if len(now_files) >= my_len:
                    f.write(f'<tr><td><div><span style="width: 2vw;">{str(iii) + "_" + str(b)}</span></div></td>')
                    for z, x in enumerate(now_files):
                        f.write(f'<td><div><img class="lazy" onerror="this.onerror=null; this.remove();" data-src="/~relh/experiments/{args.name}/outputs/{x}"></div></td>')
                    f.write('</tr>')

                future_files = sorted([x for x in files_dict[iii]['future'] if int(x.split('_')[1]) == b])
                if len(future_files) >= my_len: 
                    f.write(f'<tr><td><div><span style="width: 2vw;">{str(iii) + "_" + str(b)} +{30}</span></div></td>')
                    for z, x in enumerate(future_files):
                        f.write(f'<td><div><img class="lazy" onerror="this.onerror=null; this.remove();" data-src="/~relh/experiments/{args.name}/outputs/{x}"></div></td>')
                    f.write('</tr>')
        f.write('</tbody></table>')
        f.write('''<script>
                var lazyloadinstance = new LazyLoad({
                  // Your custom settings go here
                });
        </script>''')
        f.write('</div></body></html>')

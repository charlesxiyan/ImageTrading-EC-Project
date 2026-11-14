"""Prepare complete windows, preview images, train, and explicitly evaluate test."""
import argparse
import json
from pathlib import Path
from .data import read_config,prepare,rescreen,audit_prepared


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','rescreen','audit','preview','train','test'])
    p.add_argument('--config',default='configs/us_5min.json')
    p.add_argument('--smoke',action='store_true',help='Two train batches and a tiny validation subset; no test access')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--source',type=Path,help='rescreen: existing prepared/ folder to copy bars from')
    a=p.parse_args();config=read_config(a.config)
    if a.command=='prepare':
        prepare(config)
    elif a.command=='rescreen':
        if a.source is None:
            p.error('rescreen requires --source <existing prepared folder>')
        rescreen(config,a.source)
    elif a.command=='audit':
        audit_prepared(config)
    elif a.command=='preview':
        from PIL import Image,ImageDraw
        from .dataset import ImageDataset
        data=ImageDataset(Path(config['artifact_root'])/'prepared','train')
        root=Path(config['artifact_root'])/'preview';root.mkdir(exist_ok=True)
        records=[]
        size=3*config['lookback'];display_size=max(size,108);tile=max(size+20,150)
        sheet=Image.new('RGB',(3*tile,display_size+65),'#16191e')
        draw=ImageDraw.Draw(sheet)
        for position,index in enumerate([0,len(data)//2,len(data)-1]):
            array=data.image(index)
            image=Image.fromarray(array)
            image.save(root/f'train_{position}.png')
            x=position*tile+8
            display=image.resize((display_size,display_size),resample=Image.Resampling.NEAREST)
            sheet.paste(display.convert('RGB'),(position*tile+(tile-display_size)//2,45))
            row=data.samples.iloc[index]
            draw.text((x,8),f'PERMNO {int(row.permno)} | y={int(row.label)}',fill='white')
            records.append({k:str(v) for k,v in row.to_dict().items()})
        sheet.save(root/'contact_sheet.png')
        (root/'samples.json').write_text(json.dumps(records,indent=2)+'\n')
        print(root/'contact_sheet.png')
    else:
        from .train import train,test
        if a.command=='train':
            train(config,a.smoke)
        else:
            if a.checkpoint is None:
                p.error('test requires --checkpoint; evaluation is never automatic')
            test(config,a.checkpoint)


if __name__=='__main__':
    main()

gt = "/home/neuiva2/liweixi/data/MOT17/train/MOT17-13/gt/gt.txt"
gt_half = "/home/neuiva2/liweixi/data/MOT17/train/MOT17-13/gt/gt_half.txt"
wf = open(gt_half, 'a')
with open(gt) as rf:
    for line in rf.readlines():
        sp = line.split(',')
        img = sp[0]
        if int(img) > 750//2:
            wf.write(line)

wf.close()
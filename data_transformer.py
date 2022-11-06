import csv

# wf = open('data/train_annots_transform.csv', 'w')
# data = []
# with open('data/train_annots.csv')as f:
#
#     f_csv = csv.reader(f)
#     for row in f_csv:
#         img_path = row[0]
#         sp = img_path.split('/')
#         seq_name = sp[1]
#         img_height = 1080.0
#         img_width = 1920.0
#         if seq_name == 'MOT17-05':
#             img_height = 480.0
#             img_width = 640.0
#         ctr_x =  (float(row[2]) + float(row[4])) / 2
#         ctr_y = (float(row[3]) + float(row[5])) / 2
#         w = float(row[4]) - float(row[2])
#         h = float(row[5]) - float(row[3])
#         row[2] = round(ctr_x / img_width,5)
#         row[3] = round(ctr_y / img_height, 5)
#         row[4] = round(w / img_width, 5)
#         row[5] = round(h / img_height, 5)
#         data.append(row)
# f_csv = csv.writer(wf)
# f_csv.writerows(data)
# wf.close()

# rf = open('data/train_annots_transform.csv')
# for i in csv.reader(rf):
#     print(i)

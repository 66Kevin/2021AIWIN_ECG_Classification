import csv

from itertools import islice

csv_reader_1 = csv.reader(open("answer/answer_0.8825.csv"))
csv_reader_2 = csv.reader(open("answer/answer_0.8857.csv"))
csv_reader_3 = csv.reader(open("answer/answer_0.8866.csv"))
csv_reader_4 = csv.reader(open("answer/answer_0.8937.csv"))
csv_reader_5 = csv.reader(open("answer/answer_0.8947.csv"))

items_1 = []
items_2 = []
items_3 = []
items_4 = []
items_5 = []

for item in islice(csv_reader_1, 1, None):
    items_1.append([item[0], item[1]])

for item in islice(csv_reader_2, 1, None):
    items_2.append([item[0], item[1]])

for item in islice(csv_reader_3, 1, None):
    items_3.append([item[0], item[1]])

for item in islice(csv_reader_4, 1, None):
    items_4.append([item[0], item[1]])

for item in islice(csv_reader_5, 1, None):
    items_5.append([item[0], item[1]])

final_items = []

for i in range(len(items_1)):
    val_0 = 0
    val_1 = 0

    if int(items_1[i][1]) == 1:
        val_1 += 1
    else:
        val_0 += 1

    if int(items_2[i][1]) == 1:
        val_1 += 1
    else:
        val_0 += 1

    if int(items_3[i][1]) == 1:
        val_1 += 1
    else:
        val_0 += 1

    if int(items_4[i][1]) == 1:
        val_1 += 1
    else:
        val_0 += 1

    if int(items_5[i][1]) == 1:
        val_1 += 1
    else:
        val_0 += 1

    print(val_1, val_0)

    if val_1 > val_0:
        final_items.append([items_1[i][0], 1])
    else:
        final_items.append([items_1[i][0], 0])

print(final_items)

with open('answer.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    for item in final_items:
        writer.writerow(item)


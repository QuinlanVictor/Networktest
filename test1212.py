# 编写xml文件的测试程序
'''
制作xml文件的测试程序
单个文件标注xml的测试程序
vesion：1211编辑测试版
author：Quinlan
'''

import xlrd
import xml.dom.minidom

xls_data = xlrd.open_workbook("huizong1212quchong.xls")
table = xls_data.sheet_by_index(0)
dcmnumber = table.col_values(0)
minx = table.col_values(1)
miny = table.col_values(2)
maxx = table.col_values(3)
maxy = table.col_values(4)
minx = [int(x) for x in minx]
miny = [int(x) for x in miny]
maxx = [int(x) for x in maxx]
maxy = [int(x) for x in maxy]
dcmnumber = [int(x) for x in dcmnumber]
dcmnumber = [str(x) for x in dcmnumber]

# for num in range(len(dcmnumber)):
#    x = minx[1]
#   y = miny[1]
#   w = maxx[1] - minx[1]
#   h = maxy[1] - miny[1]

doc = xml.dom.minidom.Document()
root = doc.createElement('annotation')
doc.appendChild(root)

nodefolder = doc.createElement('folder')
nodefolder.appendChild(doc.createTextNode(str('JPEG')))

nodefilename = doc.createElement('filename')
nodefilename.appendChild(doc.createTextNode(str(dcmnumber[1] + '.jpg')))

nodepath = doc.createElement('path')
nodepath.appendChild(doc.createTextNode(str('E:')))

nodesource = doc.createElement('source')
nodedatabase = doc.createElement('databse')
nodedatabase.appendChild(doc.createTextNode(str('Quinlan')))
nodesource.appendChild(nodedatabase)

nodesize = doc.createElement('size')
nodewidth = doc.createElement('width')
nodeheight = doc.createElement('height')
nodedepth = doc.createElement('depth')
nodewidth.appendChild(doc.createTextNode(str('512')))
nodeheight.appendChild(doc.createTextNode(str('512')))
nodedepth.appendChild(doc.createTextNode(str('3')))
nodesize.appendChild(nodewidth)
nodesize.appendChild(nodeheight)
nodesize.appendChild(nodedepth)

nodesegmented = doc.createElement('segmented')
nodesegmented.appendChild(doc.createTextNode(str('0')))

nodeobject = doc.createElement('object')
nodename = doc.createElement('name')
nodename.appendChild(doc.createTextNode(str('nodle')))
nodepose = doc.createElement('pose')
nodepose.appendChild(doc.createTextNode(str('0')))
nodetruncated = doc.createElement('truncasted')
nodetruncated.appendChild(doc.createTextNode(str('0')))

nodebndbox = doc.createElement('bndbox')
nodexmin = doc.createElement('xmin')
nodexmin.appendChild(doc.createTextNode(str(minx[1])))
nodeymin = doc.createElement('ymin')
nodeymin.appendChild(doc.createTextNode(str(miny[1])))
nodexmax = doc.createElement('xmax')
nodexmax.appendChild(doc.createTextNode(str(maxx[1])))
nodeymax = doc.createElement('ymax')
nodeymax.appendChild(doc.createTextNode(str(maxy[1])))
nodebndbox.appendChild(nodexmin)
nodebndbox.appendChild(nodeymin)
nodebndbox.appendChild(nodexmax)
nodebndbox.appendChild(nodeymax)

nodeobject.appendChild(nodename)
nodeobject.appendChild(nodepose)
nodeobject.appendChild(nodetruncated)
nodeobject.appendChild(nodebndbox)

root.appendChild(nodefolder)
root.appendChild(nodefilename)
root.appendChild(nodepath)
root.appendChild(nodesource)
root.appendChild(nodesize)
root.appendChild(nodesegmented)
root.appendChild(nodeobject)

#fp = open('0002jpg.xml', 'w')
fp=open(dcmnumber[2]+'.xml','w')
doc.writexml(fp)

"""
20201125

对于调整过窗位窗宽的dicom文件预览

"""

import pydicom
import matplotlib.pyplot as plt

ct_path = 'E:\Files\Repositories\kerasYolov4/test\dicom/1.dcm'
ds=pydicom.read_file(ct_path)
img = ds.pixel_array
plt.figure()
plt.subplot(1,3,1)
plt.imshow(img, "gray")

ct_path2 = 'E:\Files\Repositories\kerasYolov4/test\dicom/tran1.dcm'
ds2=pydicom.read_file(ct_path2)
img2 = ds2.pixel_array
plt.subplot(1,3,2)
plt.imshow(img2, "gray")
#plt.show()

ct_path3 = 'E:\Files\Repositories\kerasYolov4/test\dicom/tran2.dcm'
ds3=pydicom.read_file(ct_path2)
img3 = ds3.pixel_array
plt.subplot(1,3,3)
plt.imshow(img3, "gray")
plt.show()
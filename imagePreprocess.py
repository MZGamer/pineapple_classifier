from PIL import Image

# 读取图像
image = Image.open('D:\Work\Project\School_Homework\pineapple-classifier/Pineapple_New_Test_Data/0302\cam-2\pine-bottom\img/03.JPG')

# 定义裁剪的区域
'''left = 325  # 裁剪框的左上角 x 坐标
top = 75  # 裁剪框的左上角 y 坐标
right = 575  # 裁剪框的右下角 x 坐标
bottom = 325  # 裁剪框的右下角 y 坐标'''

left = 200
top = 100
right = 800
bottom = 700
# 裁剪图像
cropped_image = image.crop((left, top, right, bottom))
cropped_image = cropped_image.convert('L')
cropped_image = cropped_image.convert("RGB")  
# 显示裁剪后的图像
cropped_image.show()
from PIL import Image
import numpy as np
import cv2
import os
import argparse

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-m",type=str,default='colight', dest='method')
    # args= parser.parse_args()
    file_path = './6x6-b/raw.PNG'
    for method in ['gcn', 'ind', 'col']:
      img = Image.open(file_path)
      # img1 = img1.convert('RGBA')
      # img2 = img2.convert('RGB')
      pixdata = img.load()
      colorset = set()
      if method == 'col':
        # 红色
        for y in range(img.size[1]):
          for x in range(img.size[0]):
            r,g,b, a = img.getpixel((x,y))
            if not ((r>129 and g<50 and b<50) or (r<100 and g<100 and b<100)):
                colorset.add(r)
                img.putpixel((x,y),(255, 255, 255,255))
        img = img.convert('RGB')
        img.save(os.path.join(os.path.dirname(file_path), "only-col.png"))
      elif method == 'gcn':
        # 紫色
        for y in range(img.size[1]):
          for x in range(img.size[0]):
            r,g,b, a = img.getpixel((x,y))
            if not ((r>140 and r<160 and g>100 and g<130 and b>130) or (r<100 and g<100 and b<100)) :
                colorset.add(r)
                img.putpixel((x,y),(255, 255, 255,255))
        img = img.convert('RGB')
        img.save(os.path.join(os.path.dirname(file_path), "only-gcn.png"))
      elif method == 'ind':
        # 蓝色
        for y in range(img.size[1]):
          for x in range(img.size[0]):
            r,g,b, a = img.getpixel((x,y))
            if not ((r<55 and g<130 and b>150) or (r<100 and g<100 and b<100)) :
                colorset.add(r)
                img.putpixel((x,y),(255, 255, 255,255))
        img = img.convert('RGB')
        img.save(os.path.join(os.path.dirname(file_path), "only-ind.png"))
      print(colorset)


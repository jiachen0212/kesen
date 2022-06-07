import os

dir_ = r'C:\Users\15974\Desktop\111'
names = os.listdir(dir_)
for name in names:
    new_name = name[:5]+name[7:] 
    os.rename(os.path.join(dir_, name), os.path.join(dir_, new_name))
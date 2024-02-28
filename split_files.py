import os
import shutil


# 复制文件
def copy_file(src_file, dst_dir):
    if not os.path.isfile(src_file):
        print("%s not exists!" % src_file)
    else:
        fpath, fname = os.path.split(src_file)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        shutil.copy(src_file, dst_dir + fname)


# 筛选文件
def split_files(
        files_path,
        save_path,
        modulo=10,
        exclude_module=0,
):
    for root, dirs, files in os.walk(files_path):
        print(root)
        # print(files)
        for file in files:  # 遍历文件
            # print(file)
            for i in file.split("."):  # 去掉后缀
                if i.find("_") != -1:
                    tmp = i.split("_")
                    number = int(tmp[-1])  # 获取文件名中的数字
                    # print(number)
                    if number % modulo == 0:  # 取模，获取指定数字的文件
                        if exclude_module > 0:  # 排除的取模
                            if number % exclude_module == 0:  # 符合被排除的取模
                                continue  # 跳过
                        filepath = root + "/" + file
                        print(filepath)
                        copy_file(filepath, save_path)  # 复制到新文件夹


if __name__ == "__main__":
    # 原文件路径
    files_path = '/Users/xin/Downloads/RAF ruler 2'
    # 保存路径
    save_path = '/Users/xin/Downloads/RAF_2/'

    split_files(
        files_path=files_path,
        save_path=save_path,
        modulo=13,
        exclude_module=10,
    )


import glob, os
from safetensors import safe_open

print("🔍 开始全盘扫描 19 个文件，寻找损坏的本体...")
path = os.path.expanduser("~/.cache/huggingface/hub/models--mistralai--Mixtral-8x7B-v0.1/snapshots/*/*.safetensors")
files = glob.glob(path)

bad_file_found = False
for f in files:
    try:
        # 尝试打开并解析文件头部
        with safe_open(f, framework="pt") as st:
            pass
    except Exception as e:
        bad_file_found = True
        real_path = os.path.realpath(f)
        print(f"\n 损坏的文件是: {f}")
        print(f" 它的隐藏本体(Blob)是: {real_path}")

        # 同时删除本体和替身
        try:
            os.remove(real_path)
            os.remove(f)
            print(" 物理删除成功！毒瘤已被彻底清除！")
        except Exception as del_e:
            print(f" 删除失败，请检查权限: {del_e}")

if not bad_file_found:
    print("✅ 扫描完成，所有文件均完好无损！")

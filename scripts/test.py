# 查看Hora库中实际定义的对象类型

import os
import sys
sys.path.append('/opt/hora')
from hora.tasks.allegro_hand_hora import AllegroHandHora

# 尝试查看asset_files_dict的内容
try:
    # 查看源代码中asset_files_dict的定义
    import inspect
    source = inspect.getsource(AllegroHandHora._create_object_asset)
    print('_create_object_asset方法源码:')
    print(source[:500])  # 打印前500个字符
except Exception as e:
    print('无法获取源码:', e)

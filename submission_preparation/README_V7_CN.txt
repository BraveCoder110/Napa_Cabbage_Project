CS7002SR V7 — 独立解压与功能验收（NEW ONLY）

该工具不会修改原 Napa_Cabbage_Project，也不会更新已存在的提交包。
使用Python 3.10+；执行 --run-demo 还需要 torch, torchvision, numpy, opencv-python, Pillow。
在 PyCharm 的 Napa_Cabbage_Project 根目录运行（假设 V6 ZIP 位于项目上一级目录）：

python.exe submission_preparation\verify_artifact_v7.py --zip "..\CS7002SR_Artifact_V6_NEW.zip" --output "..\CS7002SR_V7_Independent_Check_NEW" --run-demo

注意：--output 必须为不存在的新路径。运行后会在该目录建立 isolated/ 独立解压副本、v7_verification_summary.json、v7_verification_checks.csv 和 demo_console_v7.txt。
打开 isolated\CS7002SR_Artifact_V6_NEW\v7_demo_result_NEW\demo_report.html 查看真实模型推理结果。

V7会验证ZIP CRC、路径安全、199个白名单文件的SHA-256、所需代码与Checkpoint、20张人工GT及配对照片、两个源域115行结果，以及独立目录真实CPU Demo。
说明：本次验证仅证实已安装软件环境下独立目录可运行，不证明全新电脑能无依赖安装。D3C01L01正.jpg 是已知的20张目标GT之一；分析模型在全部20张GT上训练，因此不能视为未知样本独立测试。Demo不输出新鲜度等级，也不输出单张Dice。

制作于 2026-11-13；仅创建新文件，不覆盖旧文件。请勿上传包含私人绝对路径的完整日志到公开平台。

CS7002SR AI Model Artifact — V6 安全打包工具

作用：从你现有 Napa_Cabbage_Project 按白名单复制代码、三个Checkpoint、20个人工GT及对应照片、实验汇总、V5.1重新推理证据、Demo脚本（如已放入项目）。输出到项目外的新目录；不会移动、删除、更改旧项目文件。使用Python标准库。

步骤1：把 build_submission_artifact_v6.py 复制到 Napa_Cabbage_Project/submission_preparation/ 。
步骤2：在项目根目录先做只读预检查：
python.exe submission_preparation\build_submission_artifact_v6.py --output "..\CS7002SR_Artifact_V6_NEW" --dry-run
步骤3：若无报错，执行真正新建目录+ZIP：
python.exe submission_preparation\build_submission_artifact_v6.py --output "..\CS7002SR_Artifact_V6_NEW" --zip
若想连同完整自己拍摄的157张目标域照片打包，则步骤3末尾加 --include-target-157 。会明显增大体积，先确认Canvas文件限制。默认只附20张目标域人工GT及对应照片，另有已保存的157张分析结果表。

重要事项：
1. 公开的Bok Choy源数据图片、Label和二值Mask ZIP暂时不重分发；待原始发布页及许可来源最终核验。原始项目的dataset LICENSE文件不等于已经核实上游授权。
2. V6打包不代表模型已在独立目录验收。下一步执行V7，重点确认Demo路径和依赖。
3. 源域115张验证集复核结果已由两份上传ZIP确认：Custom Dice 0.957581 / IoU 0.918614；ResNet18 Dice 0.948380 / IoU 0.901827。
4. 这个项目尚未建立真实新鲜度分级标签；绝不可把黄色比例描述为经过验证的新鲜度百分比。
5. 如果输出目录已存在，脚本拒绝覆盖，请使用不同的新目录名。
6. 与课程有关的Technical Report和LSEP不在这里重写或更改。

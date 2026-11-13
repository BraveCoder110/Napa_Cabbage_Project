# notes：

Some models and datasets are quite large and have been uploaded to https://huggingface.co/BraveCoder110/Napa_Cabbage_Project.

## Corresponding directory：

experiments/session4_4_transfer_v1/best_resnet18_unet.pth
experiments/session6_2_analysis_roi_candidates_v1/resnet18_target_analysis_only.pth 
experiments/session4_3_smoke_v1/resnet18_unet_smoke_test.pth
E3_Session4/cifar-10-python.tar.gz
E3_Session2/Bok_Choy_Dataset.zip



# Project Structure

```text
Napa_Cabbage_Project/
├── datasets/
│   ├── Bok_Choy_Dataset/
│   │   ├── images/
│   │   │   ├── hik_0001.jpg
│   │   │   ├── hik_0002.jpg
│   │   │   ├── hik_0003.jpg
│   │   │   ├── hik_0004.jpg
│   │   │   ├── hik_0005.jpg
│   │   │   ├── hik_0006.jpg
│   │   │   ├── hik_0007.jpg
│   │   │   ├── hik_0008.jpg
│   │   │   ├── hik_0009.jpg
│   │   │   ├── hik_0010.jpg
│   │   │   ├── hik_0011.jpg
│   │   │   ├── hik_0012.jpg
│   │   │   ├── hik_0013.jpg
│   │   │   ├── hik_0014.jpg
│   │   │   ├── hik_0015.jpg
│   │   │   ├── hik_0016.jpg
│   │   │   ├── hik_0017.jpg
│   │   │   ├── hik_0018.jpg
│   │   │   ├── hik_0019.jpg
│   │   │   ├── hik_0020.jpg
│   │   │   ├── hik_0021.jpg
│   │   │   ├── hik_0022.jpg
│   │   │   ├── hik_0023.jpg
│   │   │   ├── hik_0024.jpg
│   │   │   ├── hik_0025.jpg
│   │   │   ├── hik_0026.jpg
│   │   │   ├── hik_0027.jpg
│   │   │   ├── hik_0028.jpg
│   │   │   ├── hik_0029.jpg
│   │   │   ├── hik_0030.jpg
│   │   │   ├── hik_0031.jpg
│   │   │   ├── hik_0032.jpg
│   │   │   ├── hik_0033.jpg
│   │   │   ├── hik_0034.jpg
│   │   │   ├── hik_0035.jpg
│   │   │   ├── hik_0036.jpg
│   │   │   ├── hik_0037.jpg
│   │   │   ├── hik_0038.jpg
│   │   │   ├── hik_0039.jpg
│   │   │   ├── hik_0040.jpg
│   │   │   ├── hik_0041.jpg
│   │   │   ├── hik_0042.jpg
│   │   │   ├── hik_0043.jpg
│   │   │   ├── hik_0044.jpg
│   │   │   ├── hik_0045.jpg
│   │   │   ├── hik_0046.jpg
│   │   │   ├── hik_0047.jpg
│   │   │   ├── hik_0048.jpg
│   │   │   ├── hik_0049.jpg
│   │   │   ├── hik_0050.jpg
│   │   │   ├── hik_0051.jpg
│   │   │   ├── hik_0052.jpg
│   │   │   ├── hik_0053.jpg
│   │   │   ├── hik_0054.jpg
│   │   │   ├── hik_0055.jpg
│   │   │   ├── hik_0056.jpg
│   │   │   ├── hik_0057.jpg
│   │   │   ├── hik_0058.jpg
│   │   │   ├── hik_0059.jpg
│   │   │   ├── hik_0060.jpg
│   │   │   ├── hik_0061.jpg
│   │   │   ├── hik_0062.jpg
│   │   │   ├── hik_0063.jpg
│   │   │   ├── hik_0064.jpg
│   │   │   ├── hik_0065.jpg
│   │   │   ├── hik_0066.jpg
│   │   │   ├── hik_0067.jpg
│   │   │   ├── hik_0068.jpg
│   │   │   ├── hik_0069.jpg
│   │   │   ├── hik_0070.jpg
│   │   │   ├── hik_0071.jpg
│   │   │   ├── hik_0072.jpg
│   │   │   ├── hik_0073.jpg
│   │   │   ├── hik_0074.jpg
│   │   │   ├── hik_0075.jpg
│   │   │   ├── hik_0076.jpg
│   │   │   ├── hik_0077.jpg
│   │   │   ├── hik_0078.jpg
│   │   │   ├── hik_0079.jpg
│   │   │   ├── hik_0080.jpg
│   │   │   ├── hik_0081.jpg
│   │   │   ├── hik_0082.jpg
│   │   │   ├── hik_0083.jpg
│   │   │   ├── hik_0084.jpg
│   │   │   ├── hik_0085.jpg
│   │   │   ├── hik_0086.jpg
│   │   │   ├── hik_0087.jpg
│   │   │   ├── hik_0088.jpg
│   │   │   ├── hik_0089.jpg
│   │   │   ├── hik_0090.jpg
│   │   │   ├── hik_0091.jpg
│   │   │   ├── hik_0092.jpg
│   │   │   ├── hik_0093.jpg
│   │   │   ├── hik_0094.jpg
│   │   │   ├── hik_0095.jpg
│   │   │   ├── hik_0096.jpg
│   │   │   ├── hik_0097.jpg
│   │   │   ├── hik_0098.jpg
│   │   │   ├── hik_0099.jpg
│   │   │   ├── hik_0100.jpg
│   │   │   ├── hik_0101.jpg
│   │   │   ├── hik_0102.jpg
│   │   │   ├── hik_0103.jpg
│   │   │   ├── hik_0104.jpg
│   │   │   ├── hik_0105.jpg
│   │   │   ├── hik_0106.jpg
│   │   │   ├── hik_0107.jpg
│   │   │   ├── hik_0108.jpg
│   │   │   ├── hik_0109.jpg
│   │   │   ├── hik_0110.jpg
│   │   │   ├── hik_0111.jpg
│   │   │   ├── hik_0112.jpg
│   │   │   ├── hik_0113.jpg
│   │   │   ├── hik_0114.jpg
│   │   │   ├── hik_0115.jpg
│   │   │   ├── hik_0116.jpg
│   │   │   ├── hik_0117.jpg
│   │   │   ├── hik_0118.jpg
│   │   │   ├── hik_0119.jpg
│   │   │   ├── hik_0120.jpg
│   │   │   ├── hik_0121.jpg
│   │   │   ├── hik_0122.jpg
│   │   │   ├── hik_0123.jpg
│   │   │   ├── hik_0124.jpg
│   │   │   ├── hik_0125.jpg
│   │   │   ├── hik_0126.jpg
│   │   │   ├── hik_0127.jpg
│   │   │   ├── hik_0128.jpg
│   │   │   ├── hik_0129.jpg
│   │   │   ├── hik_0130.jpg
│   │   │   ├── hik_0131.jpg
│   │   │   ├── hik_0132.jpg
│   │   │   ├── hik_0133.jpg
│   │   │   ├── hik_0134.jpg
│   │   │   ├── hik_0135.jpg
│   │   │   ├── hik_0136.jpg
│   │   │   ├── hik_0137.jpg
│   │   │   ├── hik_0138.jpg
│   │   │   ├── hik_0139.jpg
│   │   │   ├── hik_0140.jpg
│   │   │   ├── hik_0141.jpg
│   │   │   ├── hik_0142.jpg
│   │   │   ├── hik_0143.jpg
│   │   │   ├── hik_0144.jpg
│   │   │   ├── hik_0145.jpg
│   │   │   ├── hik_0146.jpg
│   │   │   ├── hik_0147.jpg
│   │   │   ├── hik_0148.jpg
│   │   │   ├── hik_0149.jpg
│   │   │   ├── hik_0150.jpg
│   │   │   ├── hik_0151.jpg
│   │   │   ├── hik_0152.jpg
│   │   │   ├── hik_0153.jpg
│   │   │   ├── hik_0154.jpg
│   │   │   ├── hik_0155.jpg
│   │   │   ├── hik_0156.jpg
│   │   │   ├── hik_0157.jpg
│   │   │   ├── hik_0158.jpg
│   │   │   ├── hik_0159.jpg
│   │   │   ├── hik_0160.jpg
│   │   │   ├── hik_0161.jpg
│   │   │   ├── hik_0162.jpg
│   │   │   ├── hik_0163.jpg
│   │   │   ├── hik_0164.jpg
│   │   │   ├── hik_0165.jpg
│   │   │   ├── hik_0166.jpg
│   │   │   ├── hik_0167.jpg
│   │   │   ├── hik_0168.jpg
│   │   │   ├── hik_0169.jpg
│   │   │   ├── hik_0170.jpg
│   │   │   ├── hik_0171.jpg
│   │   │   ├── hik_0172.jpg
│   │   │   ├── hik_0173.jpg
│   │   │   ├── hik_0174.jpg
│   │   │   ├── hik_0175.jpg
│   │   │   ├── hik_0176.jpg
│   │   │   ├── hik_0177.jpg
│   │   │   ├── hik_0178.jpg
│   │   │   ├── hik_0179.jpg
│   │   │   ├── hik_0180.jpg
│   │   │   ├── hik_0181.jpg
│   │   │   ├── hik_0182.jpg
│   │   │   ├── hik_0183.jpg
│   │   │   ├── hik_0184.jpg
│   │   │   ├── hik_0185.jpg
│   │   │   ├── hik_0186.jpg
│   │   │   ├── hik_0187.jpg
│   │   │   ├── hik_0188.jpg
│   │   │   ├── hik_0189.jpg
│   │   │   ├── hik_0190.jpg
│   │   │   ├── hik_0191.jpg
│   │   │   ├── hik_0192.jpg
│   │   │   ├── hik_0193.jpg
│   │   │   ├── hik_0194.jpg
│   │   │   ├── hik_0195.jpg
│   │   │   ├── hik_0196.jpg
│   │   │   ├── hik_0197.jpg
│   │   │   ├── hik_0198.jpg
│   │   │   ├── hik_0199.jpg
│   │   │   ├── hik_0200.jpg
│   │   │   ├── hik_0201.jpg
│   │   │   ├── hik_0202.jpg
│   │   │   ├── hik_0203.jpg
│   │   │   ├── hik_0204.jpg
│   │   │   ├── hik_0205.jpg
│   │   │   ├── hik_0206.jpg
│   │   │   ├── hik_0207.jpg
│   │   │   ├── hik_0208.jpg
│   │   │   ├── hik_0209.jpg
│   │   │   ├── hik_0210.jpg
│   │   │   ├── hik_0211.jpg
│   │   │   ├── hik_0212.jpg
│   │   │   ├── hik_0213.jpg
│   │   │   ├── hik_0214.jpg
│   │   │   ├── hik_0215.jpg
│   │   │   ├── hik_0216.jpg
│   │   │   ├── hik_0217.jpg
│   │   │   ├── hik_0218.jpg
│   │   │   ├── hik_0219.jpg
│   │   │   ├── hik_0220.jpg
│   │   │   ├── hik_0221.jpg
│   │   │   ├── hik_0222.jpg
│   │   │   ├── hik_0223.jpg
│   │   │   ├── hik_0224.jpg
│   │   │   ├── hik_0225.jpg
│   │   │   ├── hik_0226.jpg
│   │   │   ├── hik_0227.jpg
│   │   │   ├── hik_0228.jpg
│   │   │   ├── hik_0229.jpg
│   │   │   ├── hik_0230.jpg
│   │   │   ├── hik_0231.jpg
│   │   │   ├── hik_0232.jpg
│   │   │   ├── hik_0233.jpg
│   │   │   ├── hik_0234.jpg
│   │   │   ├── hik_0235.jpg
│   │   │   ├── hik_0236.jpg
│   │   │   ├── hik_0237.jpg
│   │   │   ├── hik_0238.jpg
│   │   │   ├── hik_0239.jpg
│   │   │   ├── hik_0240.jpg
│   │   │   ├── hik_0241.jpg
│   │   │   ├── hik_0242.jpg
│   │   │   ├── hik_0243.jpg
│   │   │   ├── hik_0244.jpg
│   │   │   ├── hik_0245.jpg
│   │   │   ├── hik_0246.jpg
│   │   │   ├── hik_0247.jpg
│   │   │   ├── hik_0248.jpg
│   │   │   ├── hik_0249.jpg
│   │   │   ├── hik_0250.jpg
│   │   │   ├── hik_0251.jpg
│   │   │   ├── hik_0252.jpg
│   │   │   ├── hik_0253.jpg
│   │   │   ├── hik_0254.jpg
│   │   │   ├── hik_0255.jpg
│   │   │   ├── hik_0256.jpg
│   │   │   ├── hik_0257.jpg
│   │   │   ├── hik_0258.jpg
│   │   │   ├── hik_0259.jpg
│   │   │   ├── hik_0260.jpg
│   │   │   ├── hik_0261.jpg
│   │   │   ├── hik_0262.jpg
│   │   │   ├── hik_0263.jpg
│   │   │   ├── hik_0264.jpg
│   │   │   ├── hik_0265.jpg
│   │   │   ├── hik_0266.jpg
│   │   │   ├── hik_0267.jpg
│   │   │   ├── hik_0268.jpg
│   │   │   ├── hik_0269.jpg
│   │   │   ├── hik_0270.jpg
│   │   │   ├── hik_0271.jpg
│   │   │   ├── hik_0272.jpg
│   │   │   ├── hik_0273.jpg
│   │   │   ├── hik_0274.jpg
│   │   │   ├── hik_0275.jpg
│   │   │   ├── hik_0276.jpg
│   │   │   ├── hik_0277.jpg
│   │   │   ├── hik_0278.jpg
│   │   │   ├── hik_0279.jpg
│   │   │   ├── hik_0280.jpg
│   │   │   ├── hik_0281.jpg
│   │   │   ├── hik_0282.jpg
│   │   │   ├── hik_0283.jpg
│   │   │   ├── hik_0284.jpg
│   │   │   ├── hik_0285.jpg
│   │   │   ├── hik_0286.jpg
│   │   │   ├── hik_0287.jpg
│   │   │   ├── hik_0288.jpg
│   │   │   ├── hik_0289.jpg
│   │   │   ├── hik_0290.jpg
│   │   │   ├── hik_0291.jpg
│   │   │   ├── hik_0292.jpg
│   │   │   ├── hik_0293.jpg
│   │   │   ├── hik_0294.jpg
│   │   │   ├── hik_0295.jpg
│   │   │   ├── hik_0296.jpg
│   │   │   ├── hik_0297.jpg
│   │   │   ├── hik_0298.jpg
│   │   │   ├── hik_0299.jpg
│   │   │   ├── hik_0300.jpg
│   │   │   ├── hik_0301.jpg
│   │   │   ├── hik_0302.jpg
│   │   │   ├── hik_0303.jpg
│   │   │   ├── hik_0304.jpg
│   │   │   ├── hik_0305.jpg
│   │   │   ├── hik_0306.jpg
│   │   │   ├── hik_0307.jpg
│   │   │   ├── hik_0308.jpg
│   │   │   ├── hik_0309.jpg
│   │   │   ├── hik_0310.jpg
│   │   │   ├── hik_0311.jpg
│   │   │   ├── hik_0312.jpg
│   │   │   ├── hik_0313.jpg
│   │   │   ├── hik_0314.jpg
│   │   │   ├── hik_0315.jpg
│   │   │   ├── hik_0316.jpg
│   │   │   ├── hik_0317.jpg
│   │   │   ├── hik_0318.jpg
│   │   │   ├── hik_0319.jpg
│   │   │   ├── hik_0320.jpg
│   │   │   ├── hik_0321.jpg
│   │   │   ├── hik_0322.jpg
│   │   │   ├── hik_0323.jpg
│   │   │   ├── hik_0324.jpg
│   │   │   ├── hik_0325.jpg
│   │   │   ├── hik_0326.jpg
│   │   │   ├── hik_0327.jpg
│   │   │   ├── hik_0328.jpg
│   │   │   ├── hik_0329.jpg
│   │   │   ├── hik_0330.jpg
│   │   │   ├── hik_0331.jpg
│   │   │   ├── hik_0332.jpg
│   │   │   ├── hik_0333.jpg
│   │   │   ├── hik_0334.jpg
│   │   │   ├── hik_0335.jpg
│   │   │   ├── hik_0336.jpg
│   │   │   ├── hik_0337.jpg
│   │   │   ├── hik_0338.jpg
│   │   │   ├── hik_0339.jpg
│   │   │   ├── hik_0340.jpg
│   │   │   ├── hik_0341.jpg
│   │   │   ├── hik_0342.jpg
│   │   │   ├── hik_0343.jpg
│   │   │   ├── hik_0344.jpg
│   │   │   ├── hik_0345.jpg
│   │   │   ├── hik_0346.jpg
│   │   │   ├── hik_0347.jpg
│   │   │   ├── hik_0348.jpg
│   │   │   ├── hik_0349.jpg
│   │   │   ├── hik_0350.jpg
│   │   │   ├── hik_0351.jpg
│   │   │   ├── hik_0352.jpg
│   │   │   ├── hik_0353.jpg
│   │   │   ├── hik_0354.jpg
│   │   │   ├── hik_0355.jpg
│   │   │   ├── hik_0356.jpg
│   │   │   ├── hik_0357.jpg
│   │   │   ├── hik_0358.jpg
│   │   │   ├── hik_0359.jpg
│   │   │   ├── hik_0360.jpg
│   │   │   ├── hik_0361.jpg
│   │   │   ├── hik_0362.jpg
│   │   │   ├── hik_0363.jpg
│   │   │   ├── hik_0364.jpg
│   │   │   ├── hik_0365.jpg
│   │   │   ├── hik_0366.jpg
│   │   │   ├── hik_0367.jpg
│   │   │   ├── hik_0368.jpg
│   │   │   ├── hik_0369.jpg
│   │   │   ├── hik_0370.jpg
│   │   │   ├── hik_0371.jpg
│   │   │   ├── hik_0372.jpg
│   │   │   ├── hik_0373.jpg
│   │   │   ├── hik_0374.jpg
│   │   │   ├── hik_0375.jpg
│   │   │   ├── hik_0376.jpg
│   │   │   ├── hik_0377.jpg
│   │   │   ├── hik_0378.jpg
│   │   │   ├── hik_0379.jpg
│   │   │   ├── hik_0380.jpg
│   │   │   ├── hik_0381.jpg
│   │   │   ├── hik_0382.jpg
│   │   │   ├── hik_0383.jpg
│   │   │   ├── hik_0384.jpg
│   │   │   ├── hik_0385.jpg
│   │   │   ├── hik_0386.jpg
│   │   │   ├── hik_0387.jpg
│   │   │   ├── hik_0388.jpg
│   │   │   ├── hik_0389.jpg
│   │   │   ├── hik_0390.jpg
│   │   │   ├── hik_0391.jpg
│   │   │   ├── hik_0392.jpg
│   │   │   ├── hik_0393.jpg
│   │   │   ├── hik_0394.jpg
│   │   │   ├── hik_0395.jpg
│   │   │   ├── hik_0396.jpg
│   │   │   ├── hik_0397.jpg
│   │   │   ├── hik_0398.jpg
│   │   │   ├── hik_0399.jpg
│   │   │   ├── hik_0400.jpg
│   │   │   ├── hik_0401.jpg
│   │   │   ├── hik_0402.jpg
│   │   │   ├── hik_0403.jpg
│   │   │   ├── hik_0404.jpg
│   │   │   ├── hik_0405.jpg
│   │   │   ├── hik_0406.jpg
│   │   │   ├── hik_0407.jpg
│   │   │   ├── hik_0408.jpg
│   │   │   ├── hik_0409.jpg
│   │   │   ├── hik_0410.jpg
│   │   │   ├── hik_0411.jpg
│   │   │   ├── hik_0412.jpg
│   │   │   ├── hik_0413.jpg
│   │   │   ├── hik_0414.jpg
│   │   │   ├── hik_0415.jpg
│   │   │   ├── hik_0416.jpg
│   │   │   ├── hik_0417.jpg
│   │   │   ├── hik_0418.jpg
│   │   │   ├── hik_0419.jpg
│   │   │   ├── hik_0420.jpg
│   │   │   ├── hik_0421.jpg
│   │   │   ├── hik_0422.jpg
│   │   │   ├── hik_0423.jpg
│   │   │   ├── hik_0424.jpg
│   │   │   ├── hik_0425.jpg
│   │   │   ├── hik_0426.jpg
│   │   │   ├── hik_0427.jpg
│   │   │   ├── hik_0428.jpg
│   │   │   ├── hik_0429.jpg
│   │   │   ├── hik_0430.jpg
│   │   │   ├── hik_0431.jpg
│   │   │   ├── hik_0432.jpg
│   │   │   ├── hik_0433.jpg
│   │   │   ├── hik_0434.jpg
│   │   │   ├── hik_0435.jpg
│   │   │   ├── hik_0436.jpg
│   │   │   ├── hik_0437.jpg
│   │   │   ├── hik_0438.jpg
│   │   │   ├── hik_0439.jpg
│   │   │   ├── hik_0440.jpg
│   │   │   ├── hik_0441.jpg
│   │   │   ├── hik_0442.jpg
│   │   │   ├── hik_0443.jpg
│   │   │   ├── hik_0444.jpg
│   │   │   ├── hik_0445.jpg
│   │   │   ├── hik_0446.jpg
│   │   │   ├── hik_0447.jpg
│   │   │   ├── hik_0448.jpg
│   │   │   ├── hik_0449.jpg
│   │   │   ├── hik_0450.jpg
│   │   │   ├── hik_0451.jpg
│   │   │   ├── hik_0452.jpg
│   │   │   ├── hik_0453.jpg
│   │   │   ├── hik_0454.jpg
│   │   │   ├── hik_0455.jpg
│   │   │   ├── hik_0456.jpg
│   │   │   ├── hik_0457.jpg
│   │   │   ├── hik_0458.jpg
│   │   │   ├── hik_0459.jpg
│   │   │   ├── hik_0460.jpg
│   │   │   ├── hik_0461.jpg
│   │   │   ├── hik_0462.jpg
│   │   │   ├── hik_0463.jpg
│   │   │   ├── hik_0464.jpg
│   │   │   ├── hik_0465.jpg
│   │   │   ├── hik_0466.jpg
│   │   │   ├── hik_0467.jpg
│   │   │   ├── hik_0468.jpg
│   │   │   ├── hik_0469.jpg
│   │   │   ├── hik_0470.jpg
│   │   │   ├── hik_0471.jpg
│   │   │   ├── hik_0472.jpg
│   │   │   ├── hik_0473.jpg
│   │   │   ├── hik_0474.jpg
│   │   │   ├── hik_0475.jpg
│   │   │   ├── hik_0476.jpg
│   │   │   ├── hik_0477.jpg
│   │   │   ├── hik_0478.jpg
│   │   │   ├── hik_0479.jpg
│   │   │   ├── hik_0480.jpg
│   │   │   ├── hik_0481.jpg
│   │   │   ├── hik_0482.jpg
│   │   │   ├── hik_0483.jpg
│   │   │   ├── hik_0484.jpg
│   │   │   ├── hik_0485.jpg
│   │   │   ├── hik_0486.jpg
│   │   │   ├── hik_0487.jpg
│   │   │   ├── hik_0488.jpg
│   │   │   ├── hik_0489.jpg
│   │   │   ├── hik_0490.jpg
│   │   │   ├── hik_0491.jpg
│   │   │   ├── hik_0492.jpg
│   │   │   ├── hik_0493.jpg
│   │   │   ├── hik_0494.jpg
│   │   │   ├── hik_0495.jpg
│   │   │   ├── hik_0496.jpg
│   │   │   ├── hik_0497.jpg
│   │   │   ├── hik_0498.jpg
│   │   │   ├── hik_0499.jpg
│   │   │   ├── hik_0500.jpg
│   │   │   ├── hik_0501.jpg
│   │   │   ├── hik_0502.jpg
│   │   │   ├── hik_0503.jpg
│   │   │   ├── hik_0504.jpg
│   │   │   ├── hik_0505.jpg
│   │   │   ├── hik_0506.jpg
│   │   │   ├── hik_0507.jpg
│   │   │   ├── hik_0508.jpg
│   │   │   ├── hik_0509.jpg
│   │   │   ├── hik_0510.jpg
│   │   │   ├── hik_0511.jpg
│   │   │   ├── hik_0512.jpg
│   │   │   ├── hik_0513.jpg
│   │   │   ├── hik_0514.jpg
│   │   │   ├── hik_0515.jpg
│   │   │   ├── hik_0516.jpg
│   │   │   ├── hik_0517.jpg
│   │   │   ├── hik_0518.jpg
│   │   │   ├── hik_0519.jpg
│   │   │   ├── hik_0520.jpg
│   │   │   ├── hik_0521.jpg
│   │   │   ├── hik_0522.jpg
│   │   │   ├── hik_0523.jpg
│   │   │   ├── hik_0524.jpg
│   │   │   ├── hik_0525.jpg
│   │   │   ├── hik_0526.jpg
│   │   │   ├── hik_0527.jpg
│   │   │   ├── hik_0528.jpg
│   │   │   ├── hik_0529.jpg
│   │   │   ├── hik_0530.jpg
│   │   │   ├── hik_0531.jpg
│   │   │   ├── hik_0532.jpg
│   │   │   ├── hik_0533.jpg
│   │   │   ├── hik_0534.jpg
│   │   │   ├── hik_0535.jpg
│   │   │   ├── hik_0536.jpg
│   │   │   ├── hik_0537.jpg
│   │   │   ├── hik_0538.jpg
│   │   │   ├── hik_0539.jpg
│   │   │   ├── hik_0540.jpg
│   │   │   ├── hik_0541.jpg
│   │   │   ├── hik_0542.jpg
│   │   │   ├── hik_0543.jpg
│   │   │   ├── hik_0544.jpg
│   │   │   ├── hik_0545.jpg
│   │   │   ├── hik_0546.jpg
│   │   │   ├── hik_0547.jpg
│   │   │   ├── hik_0548.jpg
│   │   │   ├── hik_0549.jpg
│   │   │   ├── hik_0550.jpg
│   │   │   ├── hik_0551.jpg
│   │   │   ├── hik_0552.jpg
│   │   │   ├── hik_0553.jpg
│   │   │   ├── hik_0554.jpg
│   │   │   ├── hik_0555.jpg
│   │   │   ├── hik_0556.jpg
│   │   │   ├── hik_0557.jpg
│   │   │   ├── hik_0558.jpg
│   │   │   ├── hik_0559.jpg
│   │   │   ├── hik_0560.jpg
│   │   │   ├── hik_0561.jpg
│   │   │   ├── hik_0562.jpg
│   │   │   ├── hik_0563.jpg
│   │   │   ├── hik_0564.jpg
│   │   │   ├── hik_0565.jpg
│   │   │   ├── hik_0566.jpg
│   │   │   ├── hik_0567.jpg
│   │   │   ├── hik_0568.jpg
│   │   │   ├── hik_0569.jpg
│   │   │   ├── hik_0570.jpg
│   │   │   ├── hik_0571.jpg
│   │   │   ├── hik_0572.jpg
│   │   │   ├── hik_0573.jpg
│   │   │   ├── hik_0574.jpg
│   │   │   └── hik_0575.jpg
│   │   ├── labels/
│   │   │   ├── hik_0001.json
│   │   │   ├── hik_0002.json
│   │   │   ├── hik_0003.json
│   │   │   ├── hik_0004.json
│   │   │   ├── hik_0005.json
│   │   │   ├── hik_0006.json
│   │   │   ├── hik_0007.json
│   │   │   ├── hik_0008.json
│   │   │   ├── hik_0009.json
│   │   │   ├── hik_0010.json
│   │   │   ├── hik_0011.json
│   │   │   ├── hik_0012.json
│   │   │   ├── hik_0013.json
│   │   │   ├── hik_0014.json
│   │   │   ├── hik_0015.json
│   │   │   ├── hik_0016.json
│   │   │   ├── hik_0017.json
│   │   │   ├── hik_0018.json
│   │   │   ├── hik_0019.json
│   │   │   ├── hik_0020.json
│   │   │   ├── hik_0021.json
│   │   │   ├── hik_0022.json
│   │   │   ├── hik_0023.json
│   │   │   ├── hik_0024.json
│   │   │   ├── hik_0025.json
│   │   │   ├── hik_0026.json
│   │   │   ├── hik_0027.json
│   │   │   ├── hik_0028.json
│   │   │   ├── hik_0029.json
│   │   │   ├── hik_0030.json
│   │   │   ├── hik_0031.json
│   │   │   ├── hik_0032.json
│   │   │   ├── hik_0033.json
│   │   │   ├── hik_0034.json
│   │   │   ├── hik_0035.json
│   │   │   ├── hik_0036.json
│   │   │   ├── hik_0037.json
│   │   │   ├── hik_0038.json
│   │   │   ├── hik_0039.json
│   │   │   ├── hik_0040.json
│   │   │   ├── hik_0041.json
│   │   │   ├── hik_0042.json
│   │   │   ├── hik_0043.json
│   │   │   ├── hik_0044.json
│   │   │   ├── hik_0045.json
│   │   │   ├── hik_0046.json
│   │   │   ├── hik_0047.json
│   │   │   ├── hik_0048.json
│   │   │   ├── hik_0049.json
│   │   │   ├── hik_0050.json
│   │   │   ├── hik_0051.json
│   │   │   ├── hik_0052.json
│   │   │   ├── hik_0053.json
│   │   │   ├── hik_0054.json
│   │   │   ├── hik_0055.json
│   │   │   ├── hik_0056.json
│   │   │   ├── hik_0057.json
│   │   │   ├── hik_0058.json
│   │   │   ├── hik_0059.json
│   │   │   ├── hik_0060.json
│   │   │   ├── hik_0061.json
│   │   │   ├── hik_0062.json
│   │   │   ├── hik_0063.json
│   │   │   ├── hik_0064.json
│   │   │   ├── hik_0065.json
│   │   │   ├── hik_0066.json
│   │   │   ├── hik_0067.json
│   │   │   ├── hik_0068.json
│   │   │   ├── hik_0069.json
│   │   │   ├── hik_0070.json
│   │   │   ├── hik_0071.json
│   │   │   ├── hik_0072.json
│   │   │   ├── hik_0073.json
│   │   │   ├── hik_0074.json
│   │   │   ├── hik_0075.json
│   │   │   ├── hik_0076.json
│   │   │   ├── hik_0077.json
│   │   │   ├── hik_0078.json
│   │   │   ├── hik_0079.json
│   │   │   ├── hik_0080.json
│   │   │   ├── hik_0081.json
│   │   │   ├── hik_0082.json
│   │   │   ├── hik_0083.json
│   │   │   ├── hik_0084.json
│   │   │   ├── hik_0085.json
│   │   │   ├── hik_0086.json
│   │   │   ├── hik_0087.json
│   │   │   ├── hik_0088.json
│   │   │   ├── hik_0089.json
│   │   │   ├── hik_0090.json
│   │   │   ├── hik_0091.json
│   │   │   ├── hik_0092.json
│   │   │   ├── hik_0093.json
│   │   │   ├── hik_0094.json
│   │   │   ├── hik_0095.json
│   │   │   ├── hik_0096.json
│   │   │   ├── hik_0097.json
│   │   │   ├── hik_0098.json
│   │   │   ├── hik_0099.json
│   │   │   ├── hik_0100.json
│   │   │   ├── hik_0101.json
│   │   │   ├── hik_0102.json
│   │   │   ├── hik_0103.json
│   │   │   ├── hik_0104.json
│   │   │   ├── hik_0105.json
│   │   │   ├── hik_0106.json
│   │   │   ├── hik_0107.json
│   │   │   ├── hik_0108.json
│   │   │   ├── hik_0109.json
│   │   │   ├── hik_0110.json
│   │   │   ├── hik_0111.json
│   │   │   ├── hik_0112.json
│   │   │   ├── hik_0113.json
│   │   │   ├── hik_0114.json
│   │   │   ├── hik_0115.json
│   │   │   ├── hik_0116.json
│   │   │   ├── hik_0117.json
│   │   │   ├── hik_0118.json
│   │   │   ├── hik_0119.json
│   │   │   ├── hik_0120.json
│   │   │   ├── hik_0121.json
│   │   │   ├── hik_0122.json
│   │   │   ├── hik_0123.json
│   │   │   ├── hik_0124.json
│   │   │   ├── hik_0125.json
│   │   │   ├── hik_0126.json
│   │   │   ├── hik_0127.json
│   │   │   ├── hik_0128.json
│   │   │   ├── hik_0129.json
│   │   │   ├── hik_0130.json
│   │   │   ├── hik_0131.json
│   │   │   ├── hik_0132.json
│   │   │   ├── hik_0133.json
│   │   │   ├── hik_0134.json
│   │   │   ├── hik_0135.json
│   │   │   ├── hik_0136.json
│   │   │   ├── hik_0137.json
│   │   │   ├── hik_0138.json
│   │   │   ├── hik_0139.json
│   │   │   ├── hik_0140.json
│   │   │   ├── hik_0141.json
│   │   │   ├── hik_0142.json
│   │   │   ├── hik_0143.json
│   │   │   ├── hik_0144.json
│   │   │   ├── hik_0145.json
│   │   │   ├── hik_0146.json
│   │   │   ├── hik_0147.json
│   │   │   ├── hik_0148.json
│   │   │   ├── hik_0149.json
│   │   │   ├── hik_0150.json
│   │   │   ├── hik_0151.json
│   │   │   ├── hik_0152.json
│   │   │   ├── hik_0153.json
│   │   │   ├── hik_0154.json
│   │   │   ├── hik_0155.json
│   │   │   ├── hik_0156.json
│   │   │   ├── hik_0157.json
│   │   │   ├── hik_0158.json
│   │   │   ├── hik_0159.json
│   │   │   ├── hik_0160.json
│   │   │   ├── hik_0161.json
│   │   │   ├── hik_0162.json
│   │   │   ├── hik_0163.json
│   │   │   ├── hik_0164.json
│   │   │   ├── hik_0165.json
│   │   │   ├── hik_0166.json
│   │   │   ├── hik_0167.json
│   │   │   ├── hik_0168.json
│   │   │   ├── hik_0169.json
│   │   │   ├── hik_0170.json
│   │   │   ├── hik_0171.json
│   │   │   ├── hik_0172.json
│   │   │   ├── hik_0173.json
│   │   │   ├── hik_0174.json
│   │   │   ├── hik_0175.json
│   │   │   ├── hik_0176.json
│   │   │   ├── hik_0177.json
│   │   │   ├── hik_0178.json
│   │   │   ├── hik_0179.json
│   │   │   ├── hik_0180.json
│   │   │   ├── hik_0181.json
│   │   │   ├── hik_0182.json
│   │   │   ├── hik_0183.json
│   │   │   ├── hik_0184.json
│   │   │   ├── hik_0185.json
│   │   │   ├── hik_0186.json
│   │   │   ├── hik_0187.json
│   │   │   ├── hik_0188.json
│   │   │   ├── hik_0189.json
│   │   │   ├── hik_0190.json
│   │   │   ├── hik_0191.json
│   │   │   ├── hik_0192.json
│   │   │   ├── hik_0193.json
│   │   │   ├── hik_0194.json
│   │   │   ├── hik_0195.json
│   │   │   ├── hik_0196.json
│   │   │   ├── hik_0197.json
│   │   │   ├── hik_0198.json
│   │   │   ├── hik_0199.json
│   │   │   ├── hik_0200.json
│   │   │   ├── hik_0201.json
│   │   │   ├── hik_0202.json
│   │   │   ├── hik_0203.json
│   │   │   ├── hik_0204.json
│   │   │   ├── hik_0205.json
│   │   │   ├── hik_0206.json
│   │   │   ├── hik_0207.json
│   │   │   ├── hik_0208.json
│   │   │   ├── hik_0209.json
│   │   │   ├── hik_0210.json
│   │   │   ├── hik_0211.json
│   │   │   ├── hik_0212.json
│   │   │   ├── hik_0213.json
│   │   │   ├── hik_0214.json
│   │   │   ├── hik_0215.json
│   │   │   ├── hik_0216.json
│   │   │   ├── hik_0217.json
│   │   │   ├── hik_0218.json
│   │   │   ├── hik_0219.json
│   │   │   ├── hik_0220.json
│   │   │   ├── hik_0221.json
│   │   │   ├── hik_0222.json
│   │   │   ├── hik_0223.json
│   │   │   ├── hik_0224.json
│   │   │   ├── hik_0225.json
│   │   │   ├── hik_0226.json
│   │   │   ├── hik_0227.json
│   │   │   ├── hik_0228.json
│   │   │   ├── hik_0229.json
│   │   │   ├── hik_0230.json
│   │   │   ├── hik_0231.json
│   │   │   ├── hik_0232.json
│   │   │   ├── hik_0233.json
│   │   │   ├── hik_0234.json
│   │   │   ├── hik_0235.json
│   │   │   ├── hik_0236.json
│   │   │   ├── hik_0237.json
│   │   │   ├── hik_0238.json
│   │   │   ├── hik_0239.json
│   │   │   ├── hik_0240.json
│   │   │   ├── hik_0241.json
│   │   │   ├── hik_0242.json
│   │   │   ├── hik_0243.json
│   │   │   ├── hik_0244.json
│   │   │   ├── hik_0245.json
│   │   │   ├── hik_0246.json
│   │   │   ├── hik_0247.json
│   │   │   ├── hik_0248.json
│   │   │   ├── hik_0249.json
│   │   │   ├── hik_0250.json
│   │   │   ├── hik_0251.json
│   │   │   ├── hik_0252.json
│   │   │   ├── hik_0253.json
│   │   │   ├── hik_0254.json
│   │   │   ├── hik_0255.json
│   │   │   ├── hik_0256.json
│   │   │   ├── hik_0257.json
│   │   │   ├── hik_0258.json
│   │   │   ├── hik_0259.json
│   │   │   ├── hik_0260.json
│   │   │   ├── hik_0261.json
│   │   │   ├── hik_0262.json
│   │   │   ├── hik_0263.json
│   │   │   ├── hik_0264.json
│   │   │   ├── hik_0265.json
│   │   │   ├── hik_0266.json
│   │   │   ├── hik_0267.json
│   │   │   ├── hik_0268.json
│   │   │   ├── hik_0269.json
│   │   │   ├── hik_0270.json
│   │   │   ├── hik_0271.json
│   │   │   ├── hik_0272.json
│   │   │   ├── hik_0273.json
│   │   │   ├── hik_0274.json
│   │   │   ├── hik_0275.json
│   │   │   ├── hik_0276.json
│   │   │   ├── hik_0277.json
│   │   │   ├── hik_0278.json
│   │   │   ├── hik_0279.json
│   │   │   ├── hik_0280.json
│   │   │   ├── hik_0281.json
│   │   │   ├── hik_0282.json
│   │   │   ├── hik_0283.json
│   │   │   ├── hik_0284.json
│   │   │   ├── hik_0285.json
│   │   │   ├── hik_0286.json
│   │   │   ├── hik_0287.json
│   │   │   ├── hik_0288.json
│   │   │   ├── hik_0289.json
│   │   │   ├── hik_0290.json
│   │   │   ├── hik_0291.json
│   │   │   ├── hik_0292.json
│   │   │   ├── hik_0293.json
│   │   │   ├── hik_0294.json
│   │   │   ├── hik_0295.json
│   │   │   ├── hik_0296.json
│   │   │   ├── hik_0297.json
│   │   │   ├── hik_0298.json
│   │   │   ├── hik_0299.json
│   │   │   ├── hik_0300.json
│   │   │   ├── hik_0301.json
│   │   │   ├── hik_0302.json
│   │   │   ├── hik_0303.json
│   │   │   ├── hik_0304.json
│   │   │   ├── hik_0305.json
│   │   │   ├── hik_0306.json
│   │   │   ├── hik_0307.json
│   │   │   ├── hik_0308.json
│   │   │   ├── hik_0309.json
│   │   │   ├── hik_0310.json
│   │   │   ├── hik_0311.json
│   │   │   ├── hik_0312.json
│   │   │   ├── hik_0313.json
│   │   │   ├── hik_0314.json
│   │   │   ├── hik_0315.json
│   │   │   ├── hik_0316.json
│   │   │   ├── hik_0317.json
│   │   │   ├── hik_0318.json
│   │   │   ├── hik_0319.json
│   │   │   ├── hik_0320.json
│   │   │   ├── hik_0321.json
│   │   │   ├── hik_0322.json
│   │   │   ├── hik_0323.json
│   │   │   ├── hik_0324.json
│   │   │   ├── hik_0325.json
│   │   │   ├── hik_0326.json
│   │   │   ├── hik_0327.json
│   │   │   ├── hik_0328.json
│   │   │   ├── hik_0329.json
│   │   │   ├── hik_0330.json
│   │   │   ├── hik_0331.json
│   │   │   ├── hik_0332.json
│   │   │   ├── hik_0333.json
│   │   │   ├── hik_0334.json
│   │   │   ├── hik_0335.json
│   │   │   ├── hik_0336.json
│   │   │   ├── hik_0337.json
│   │   │   ├── hik_0338.json
│   │   │   ├── hik_0339.json
│   │   │   ├── hik_0340.json
│   │   │   ├── hik_0341.json
│   │   │   ├── hik_0342.json
│   │   │   ├── hik_0343.json
│   │   │   ├── hik_0344.json
│   │   │   ├── hik_0345.json
│   │   │   ├── hik_0346.json
│   │   │   ├── hik_0347.json
│   │   │   ├── hik_0348.json
│   │   │   ├── hik_0349.json
│   │   │   ├── hik_0350.json
│   │   │   ├── hik_0351.json
│   │   │   ├── hik_0352.json
│   │   │   ├── hik_0353.json
│   │   │   ├── hik_0354.json
│   │   │   ├── hik_0355.json
│   │   │   ├── hik_0356.json
│   │   │   ├── hik_0357.json
│   │   │   ├── hik_0358.json
│   │   │   ├── hik_0359.json
│   │   │   ├── hik_0360.json
│   │   │   ├── hik_0361.json
│   │   │   ├── hik_0362.json
│   │   │   ├── hik_0363.json
│   │   │   ├── hik_0364.json
│   │   │   ├── hik_0365.json
│   │   │   ├── hik_0366.json
│   │   │   ├── hik_0367.json
│   │   │   ├── hik_0368.json
│   │   │   ├── hik_0369.json
│   │   │   ├── hik_0370.json
│   │   │   ├── hik_0371.json
│   │   │   ├── hik_0372.json
│   │   │   ├── hik_0373.json
│   │   │   ├── hik_0374.json
│   │   │   ├── hik_0375.json
│   │   │   ├── hik_0376.json
│   │   │   ├── hik_0377.json
│   │   │   ├── hik_0378.json
│   │   │   ├── hik_0379.json
│   │   │   ├── hik_0380.json
│   │   │   ├── hik_0381.json
│   │   │   ├── hik_0382.json
│   │   │   ├── hik_0383.json
│   │   │   ├── hik_0384.json
│   │   │   ├── hik_0385.json
│   │   │   ├── hik_0386.json
│   │   │   ├── hik_0387.json
│   │   │   ├── hik_0388.json
│   │   │   ├── hik_0389.json
│   │   │   ├── hik_0390.json
│   │   │   ├── hik_0391.json
│   │   │   ├── hik_0392.json
│   │   │   ├── hik_0393.json
│   │   │   ├── hik_0394.json
│   │   │   ├── hik_0395.json
│   │   │   ├── hik_0396.json
│   │   │   ├── hik_0397.json
│   │   │   ├── hik_0398.json
│   │   │   ├── hik_0399.json
│   │   │   ├── hik_0400.json
│   │   │   ├── hik_0401.json
│   │   │   ├── hik_0402.json
│   │   │   ├── hik_0403.json
│   │   │   ├── hik_0404.json
│   │   │   ├── hik_0405.json
│   │   │   ├── hik_0406.json
│   │   │   ├── hik_0407.json
│   │   │   ├── hik_0408.json
│   │   │   ├── hik_0409.json
│   │   │   ├── hik_0410.json
│   │   │   ├── hik_0411.json
│   │   │   ├── hik_0412.json
│   │   │   ├── hik_0413.json
│   │   │   ├── hik_0414.json
│   │   │   ├── hik_0415.json
│   │   │   ├── hik_0416.json
│   │   │   ├── hik_0417.json
│   │   │   ├── hik_0418.json
│   │   │   ├── hik_0419.json
│   │   │   ├── hik_0420.json
│   │   │   ├── hik_0421.json
│   │   │   ├── hik_0422.json
│   │   │   ├── hik_0423.json
│   │   │   ├── hik_0424.json
│   │   │   ├── hik_0425.json
│   │   │   ├── hik_0426.json
│   │   │   ├── hik_0427.json
│   │   │   ├── hik_0428.json
│   │   │   ├── hik_0429.json
│   │   │   ├── hik_0430.json
│   │   │   ├── hik_0431.json
│   │   │   ├── hik_0432.json
│   │   │   ├── hik_0433.json
│   │   │   ├── hik_0434.json
│   │   │   ├── hik_0435.json
│   │   │   ├── hik_0436.json
│   │   │   ├── hik_0437.json
│   │   │   ├── hik_0438.json
│   │   │   ├── hik_0439.json
│   │   │   ├── hik_0440.json
│   │   │   ├── hik_0441.json
│   │   │   ├── hik_0442.json
│   │   │   ├── hik_0443.json
│   │   │   ├── hik_0444.json
│   │   │   ├── hik_0445.json
│   │   │   ├── hik_0446.json
│   │   │   ├── hik_0447.json
│   │   │   ├── hik_0448.json
│   │   │   ├── hik_0449.json
│   │   │   ├── hik_0450.json
│   │   │   ├── hik_0451.json
│   │   │   ├── hik_0452.json
│   │   │   ├── hik_0453.json
│   │   │   ├── hik_0454.json
│   │   │   ├── hik_0455.json
│   │   │   ├── hik_0456.json
│   │   │   ├── hik_0457.json
│   │   │   ├── hik_0458.json
│   │   │   ├── hik_0459.json
│   │   │   ├── hik_0460.json
│   │   │   ├── hik_0461.json
│   │   │   ├── hik_0462.json
│   │   │   ├── hik_0463.json
│   │   │   ├── hik_0464.json
│   │   │   ├── hik_0465.json
│   │   │   ├── hik_0466.json
│   │   │   ├── hik_0467.json
│   │   │   ├── hik_0468.json
│   │   │   ├── hik_0469.json
│   │   │   ├── hik_0470.json
│   │   │   ├── hik_0471.json
│   │   │   ├── hik_0472.json
│   │   │   ├── hik_0473.json
│   │   │   ├── hik_0474.json
│   │   │   ├── hik_0475.json
│   │   │   ├── hik_0476.json
│   │   │   ├── hik_0477.json
│   │   │   ├── hik_0478.json
│   │   │   ├── hik_0479.json
│   │   │   ├── hik_0480.json
│   │   │   ├── hik_0481.json
│   │   │   ├── hik_0482.json
│   │   │   ├── hik_0483.json
│   │   │   ├── hik_0484.json
│   │   │   ├── hik_0485.json
│   │   │   ├── hik_0486.json
│   │   │   ├── hik_0487.json
│   │   │   ├── hik_0488.json
│   │   │   ├── hik_0489.json
│   │   │   ├── hik_0490.json
│   │   │   ├── hik_0491.json
│   │   │   ├── hik_0492.json
│   │   │   ├── hik_0493.json
│   │   │   ├── hik_0494.json
│   │   │   ├── hik_0495.json
│   │   │   ├── hik_0496.json
│   │   │   ├── hik_0497.json
│   │   │   ├── hik_0498.json
│   │   │   ├── hik_0499.json
│   │   │   ├── hik_0500.json
│   │   │   ├── hik_0501.json
│   │   │   ├── hik_0502.json
│   │   │   ├── hik_0503.json
│   │   │   ├── hik_0504.json
│   │   │   ├── hik_0505.json
│   │   │   ├── hik_0506.json
│   │   │   ├── hik_0507.json
│   │   │   ├── hik_0508.json
│   │   │   ├── hik_0509.json
│   │   │   ├── hik_0510.json
│   │   │   ├── hik_0511.json
│   │   │   ├── hik_0512.json
│   │   │   ├── hik_0513.json
│   │   │   ├── hik_0514.json
│   │   │   ├── hik_0515.json
│   │   │   ├── hik_0516.json
│   │   │   ├── hik_0517.json
│   │   │   ├── hik_0518.json
│   │   │   ├── hik_0519.json
│   │   │   ├── hik_0520.json
│   │   │   ├── hik_0521.json
│   │   │   ├── hik_0522.json
│   │   │   ├── hik_0523.json
│   │   │   ├── hik_0524.json
│   │   │   ├── hik_0525.json
│   │   │   ├── hik_0526.json
│   │   │   ├── hik_0527.json
│   │   │   ├── hik_0528.json
│   │   │   ├── hik_0529.json
│   │   │   ├── hik_0530.json
│   │   │   ├── hik_0531.json
│   │   │   ├── hik_0532.json
│   │   │   ├── hik_0533.json
│   │   │   ├── hik_0534.json
│   │   │   ├── hik_0535.json
│   │   │   ├── hik_0536.json
│   │   │   ├── hik_0537.json
│   │   │   ├── hik_0538.json
│   │   │   ├── hik_0539.json
│   │   │   ├── hik_0540.json
│   │   │   ├── hik_0541.json
│   │   │   ├── hik_0542.json
│   │   │   ├── hik_0543.json
│   │   │   ├── hik_0544.json
│   │   │   ├── hik_0545.json
│   │   │   ├── hik_0546.json
│   │   │   ├── hik_0547.json
│   │   │   ├── hik_0548.json
│   │   │   ├── hik_0549.json
│   │   │   ├── hik_0550.json
│   │   │   ├── hik_0551.json
│   │   │   ├── hik_0552.json
│   │   │   ├── hik_0553.json
│   │   │   ├── hik_0554.json
│   │   │   ├── hik_0555.json
│   │   │   ├── hik_0556.json
│   │   │   ├── hik_0557.json
│   │   │   ├── hik_0558.json
│   │   │   ├── hik_0559.json
│   │   │   ├── hik_0560.json
│   │   │   ├── hik_0561.json
│   │   │   ├── hik_0562.json
│   │   │   ├── hik_0563.json
│   │   │   ├── hik_0564.json
│   │   │   ├── hik_0565.json
│   │   │   ├── hik_0566.json
│   │   │   ├── hik_0567.json
│   │   │   ├── hik_0568.json
│   │   │   ├── hik_0569.json
│   │   │   ├── hik_0570.json
│   │   │   ├── hik_0571.json
│   │   │   ├── hik_0572.json
│   │   │   ├── hik_0573.json
│   │   │   ├── hik_0574.json
│   │   │   └── hik_0575.json
│   │   ├── checksums.sha256
│   │   ├── dataset_info.json
│   │   ├── LICENSE.txt
│   │   ├── README.md
│   │   ├── train.json
│   │   └── val.json
│   └── Huang_Jing_Bai_Cai/
│       ├── Huang_Jing_Bai_Cai_D0/
│       │   ├── C01L01反.jpg
│       │   ├── C01L01正.jpg
│       │   ├── C01L02反.jpg
│       │   ├── C01L02正.jpg
│       │   ├── C01L03反.jpg
│       │   ├── C01L03正.jpg
│       │   ├── C01L04反.jpg
│       │   ├── C01L04正.jpg
│       │   ├── C02L01反.jpg
│       │   ├── C02L01正.jpg
│       │   ├── C02L02反.jpg
│       │   ├── C02L02正.jpg
│       │   ├── C02L03反.jpg
│       │   ├── C02L03正.jpg
│       │   ├── C02L04反.jpg
│       │   ├── C02L04正.jpg
│       │   ├── C03L01反.jpg
│       │   ├── C03L01正.jpg
│       │   ├── C03L02反.jpg
│       │   ├── C03L02正.jpg
│       │   ├── C03L03反.jpg
│       │   ├── C03L03正.jpg
│       │   ├── C03L04反.jpg
│       │   ├── C03L04正.jpg
│       │   ├── C04L01反.jpg
│       │   ├── C04L01正.jpg
│       │   ├── C04L02反.jpg
│       │   ├── C04L02正.jpg
│       │   ├── C04L03正.jpg
│       │   ├── C04L04反.jpg
│       │   ├── C04L04正.jpg
│       │   ├── C05L01反.jpg
│       │   ├── C05L01正.jpg
│       │   ├── C05L02反.jpg
│       │   ├── C05L02正.jpg
│       │   ├── C05L03正.jpg
│       │   └── C05L04正.jpg
│       ├── Huang_Jing_Bai_Cai_D3/
│       │   ├── D3C01L01反.jpg
│       │   ├── D3C01L01正.jpg
│       │   ├── D3C01L02反.jpg
│       │   ├── D3C01L02正.jpg
│       │   ├── D3C01L03反.jpg
│       │   ├── D3C01L03正.jpg
│       │   ├── D3C01L04反.jpg
│       │   ├── D3C01L04正.jpg
│       │   ├── D3C02L01反.jpg
│       │   ├── D3C02L01正.jpg
│       │   ├── D3C02L02反.jpg
│       │   ├── D3C02L02正.jpg
│       │   ├── D3C02L03反.jpg
│       │   ├── D3C02L03正.jpg
│       │   ├── D3C02L04反.jpg
│       │   ├── D3C02L04正.jpg
│       │   ├── D3C03L01反.jpg
│       │   ├── D3C03L01正.jpg
│       │   ├── D3C03L02反.jpg
│       │   ├── D3C03L02正.jpg
│       │   ├── D3C03L03反.jpg
│       │   ├── D3C03L03正.jpg
│       │   ├── D3C03L04反.jpg
│       │   ├── D3C03L04正.jpg
│       │   ├── D3C04L01反.jpg
│       │   ├── D3C04L01正.jpg
│       │   ├── D3C04L02反.jpg
│       │   ├── D3C04L02正.jpg
│       │   ├── D3C04L03反.jpg
│       │   ├── D3C04L03正.jpg
│       │   ├── D3C04L04反.jpg
│       │   ├── D3C04L04正.jpg
│       │   ├── D3C05L01反.jpg
│       │   ├── D3C05L01正.jpg
│       │   ├── D3C05L02反.jpg
│       │   ├── D3C05L02正.jpg
│       │   ├── D3C05L03反.jpg
│       │   ├── D3C05L03正.jpg
│       │   ├── D3C05L04反.jpg
│       │   └── D3C05L04正.jpg
│       ├── Huang_Jing_Bai_Cai_D5/
│       │   ├── D5C01L01反.jpg
│       │   ├── D5C01L01正.jpg
│       │   ├── D5C01L02反.jpg
│       │   ├── D5C01L02正.jpg
│       │   ├── D5C01L03反.jpg
│       │   ├── D5C01L03正.jpg
│       │   ├── D5C01L04反.jpg
│       │   ├── D5C01L04正.jpg
│       │   ├── D5C02L01反.jpg
│       │   ├── D5C02L01正.jpg
│       │   ├── D5C02L02反.jpg
│       │   ├── D5C02L02正.jpg
│       │   ├── D5C02L03反.jpg
│       │   ├── D5C02L03正.jpg
│       │   ├── D5C02L04反.jpg
│       │   ├── D5C02L04正.jpg
│       │   ├── D5C03L01反.jpg
│       │   ├── D5C03L01正.jpg
│       │   ├── D5C03L02反.jpg
│       │   ├── D5C03L02正.jpg
│       │   ├── D5C03L03反.jpg
│       │   ├── D5C03L03正.jpg
│       │   ├── D5C03L04反.jpg
│       │   ├── D5C03L04正.jpg
│       │   ├── D5C04L01反.jpg
│       │   ├── D5C04L01正.jpg
│       │   ├── D5C04L02反.jpg
│       │   ├── D5C04L02正.jpg
│       │   ├── D5C04L03反.jpg
│       │   ├── D5C04L03正.jpg
│       │   ├── D5C04L04反.jpg
│       │   ├── D5C04L04正.jpg
│       │   ├── D5C05L01反.jpg
│       │   ├── D5C05L01正.jpg
│       │   ├── D5C05L02反.jpg
│       │   ├── D5C05L02正.jpg
│       │   ├── D5C05L03反.jpg
│       │   ├── D5C05L03正.jpg
│       │   ├── D5C05L04反.jpg
│       │   └── D5C05L04正.jpg
│       ├── Huang_Jing_Bai_Cai_D7/
│       │   ├── D7C01L01反.jpg
│       │   ├── D7C01L01正.jpg
│       │   ├── D7C01L02反.jpg
│       │   ├── D7C01L02正.jpg
│       │   ├── D7C01L03反.jpg
│       │   ├── D7C01L03正.jpg
│       │   ├── D7C01L04反.jpg
│       │   ├── D7C01L04正.jpg
│       │   ├── D7C02L01反.jpg
│       │   ├── D7C02L01正.jpg
│       │   ├── D7C02L02反.jpg
│       │   ├── D7C02L02正.jpg
│       │   ├── D7C02L03反.jpg
│       │   ├── D7C02L03正.jpg
│       │   ├── D7C02L04反.jpg
│       │   ├── D7C02L04正.jpg
│       │   ├── D7C03L01反.jpg
│       │   ├── D7C03L01正.jpg
│       │   ├── D7C03L02反.jpg
│       │   ├── D7C03L02正.jpg
│       │   ├── D7C03L03反.jpg
│       │   ├── D7C03L03正.jpg
│       │   ├── D7C03L04反.jpg
│       │   ├── D7C03L04正.jpg
│       │   ├── D7C04L01反.jpg
│       │   ├── D7C04L01正.jpg
│       │   ├── D7C04L02反.jpg
│       │   ├── D7C04L02正.jpg
│       │   ├── D7C04L03反.jpg
│       │   ├── D7C04L03正.jpg
│       │   ├── D7C04L04反.jpg
│       │   ├── D7C04L04正.jpg
│       │   ├── D7C05L01反.jpg
│       │   ├── D7C05L01正.jpg
│       │   ├── D7C05L02反.jpg
│       │   ├── D7C05L02正.jpg
│       │   ├── D7C05L03反.jpg
│       │   ├── D7C05L03正.jpg
│       │   ├── D7C05L04反.jpg
│       │   └── D7C05L04正.jpg
│       ├── sd-10/
│       │   ├── C01-右侧.jpg
│       │   ├── C01-左侧.jpg
│       │   ├── C01-正面.jpg
│       │   ├── C01-背面.jpg
│       │   ├── C01-顶部.jpg
│       │   ├── C02-右侧.jpg
│       │   ├── C02-左侧.jpg
│       │   ├── C02-正面.jpg
│       │   ├── C02-背面.jpg
│       │   ├── C02-顶部.jpg
│       │   ├── C03-右侧.jpg
│       │   ├── C03-左侧.jpg
│       │   ├── C03-正面.jpg
│       │   ├── C03-背面.jpg
│       │   ├── C03-顶部.jpg
│       │   ├── C04-右侧.jpg
│       │   ├── C04-左侧.jpg
│       │   ├── C04-正面.jpg
│       │   ├── C04-背面.jpg
│       │   ├── C04-顶部.jpg
│       │   ├── C05-右侧.jpg
│       │   ├── C05-左侧.jpg
│       │   ├── C05-正面.jpg
│       │   ├── C05-背面.jpg
│       │   ├── C05-顶部.jpg
│       │   └── readme.txt
│       ├── D0_D3_pairing_manifest.csv
│       ├── D3_photo_manifest.csv
│       ├── D5_photo_manifest.csv
│       ├── Huang_Jing_Bai_Cai_D0.zip
│       ├── Huang_Jing_Bai_Cai_D3.zip
│       ├── Huang_Jing_Bai_Cai_D5.zip
│       └── Huang_Jing_Bai_Cai_D7.zip
├── E3_output/
│   ├── binary_masks.zip
│   ├── issues.txt
│   ├── mask_previews.jpg
│   └── mask_quality_report.csv
├── E3_Session2/
│   ├── results/
│   │   ├── issues.txt
│   │   ├── mask_previews.jpg
│   │   └── mask_quality_report.csv
│   ├── Bok_Choy_Dataset.zip
│   ├── check_dataloader.py
│   ├── convert_polygons.py
│   └── README.txt
├── E3_Session3/
│   ├── models/
│   │   └── unet.py
│   ├── analyze_errors.py
│   ├── evaluate_baseline.py
│   ├── prepare_annotation_audit.py
│   ├── README.txt
│   ├── README_Session3_3.txt
│   ├── test_unet.py
│   ├── train_baseline.py
│   └── train_smoke_test.py
├── E3_Session4/
│   ├── cifar-10-batches-py/
│   │   ├── batches.meta
│   │   ├── data_batch_1
│   │   ├── data_batch_2
│   │   ├── data_batch_3
│   │   ├── data_batch_4
│   │   ├── data_batch_5
│   │   ├── readme.html
│   │   └── test_batch
│   ├── models/
│   │   └── resnet18_unet.py
│   ├── AAI_2.ipynb
│   ├── check_transfer_environment.py
│   ├── cifar-10-python.tar.gz
│   ├── compare_models.py
│   ├── evaluate_transfer_baseline.py
│   ├── README_Session4_3.txt
│   ├── README_Session4_4.txt
│   ├── README_Session4_5.txt
│   ├── test_resnet18_unet.py
│   ├── train_transfer_baseline.py
│   └── train_transfer_smoke_test.py
├── E3_Session5/
│   ├── diagnose_zero_shot.py
│   ├── evaluate_zero_shot.py
│   ├── prepare_target_pilot.py
│   ├── README_Session5_1.txt
│   ├── README_Session5_2.txt
│   ├── README_Session5_2_1.txt
│   ├── README_Session5_3.txt
│   └── run_grouped_few_shot.py
├── E3_Session6/
│   ├── audit_longitudinal.py
│   ├── compare_gt_auto_roi_features.py
│   ├── compare_gt_auto_roi_features_v2.py
│   ├── diagnose_photo_color_v1.py
│   ├── extract_visual_freshness_features.py
│   ├── README_Session6_1.txt
│   ├── README_Session6_2.txt
│   ├── README_Session6_3.txt
│   ├── README_Session6_4.txt
│   ├── README_Session6_4_2.txt
│   ├── README_Session6_4_Filename_Fix_v2.txt
│   └── train_final_roi_model.py
├── experiments/
│   ├── session3_3/
│   │   ├── training_log.csv
│   │   └── unet_smoke_test.pth
│   ├── session3_4_baseline_v1/
│   │   └── best_unet.pth/
│   ├── session3_4_baseline_v2/
│   │   ├── best_unet.pth
│   │   ├── config.json
│   │   ├── history.csv
│   │   └── runtime.txt
│   ├── session3_4_diagnostic/
│   │   ├── best_unet.pth
│   │   ├── config.json
│   │   ├── history.csv
│   │   └── runtime.txt
│   ├── session3_4_evaluation_v1/
│   ├── session3_4_evaluation_v2/
│   │   ├── per_image_metrics.csv
│   │   ├── predictions_preview.png
│   │   └── summary.txt
│   ├── session3_5_annotation_audit_v1/
│   │   ├── cases/
│   │   │   ├── hik_0545/
│   │   │   │   ├── original_rgb.jpg
│   │   │   │   └── source_gt_mask.png
│   │   │   ├── hik_0550/
│   │   │   │   ├── original_rgb.jpg
│   │   │   │   └── source_gt_mask.png
│   │   │   ├── hik_0553/
│   │   │   │   ├── original_rgb.jpg
│   │   │   │   └── source_gt_mask.png
│   │   │   ├── hik_0556/
│   │   │   │   ├── original_rgb.jpg
│   │   │   │   └── source_gt_mask.png
│   │   │   ├── hik_0564/
│   │   │   │   ├── original_rgb.jpg
│   │   │   │   └── source_gt_mask.png
│   │   │   ├── hik_0566/
│   │   │   │   ├── original_rgb.jpg
│   │   │   │   └── source_gt_mask.png
│   │   │   ├── hik_0567/
│   │   │   │   ├── original_rgb.jpg
│   │   │   │   └── source_gt_mask.png
│   │   │   ├── hik_0569/
│   │   │   │   ├── original_rgb.jpg
│   │   │   │   └── source_gt_mask.png
│   │   │   ├── hik_0574/
│   │   │   │   ├── original_rgb.jpg
│   │   │   │   └── source_gt_mask.png
│   │   │   └── hik_0575/
│   │   │       ├── original_rgb.jpg
│   │   │       └── source_gt_mask.png
│   │   ├── annotation_audit.csv
│   │   ├── blind_rgb_gt_overview.png
│   │   └── source_provenance.json
│   ├── session3_5_error_analysis_v1/
│   │   ├── worst_cases/
│   │   │   ├── 01_hik_0567.png
│   │   │   ├── 02_hik_0575.png
│   │   │   ├── 03_hik_0569.png
│   │   │   ├── 04_hik_0553.png
│   │   │   ├── 05_hik_0550.png
│   │   │   ├── 06_hik_0545.png
│   │   │   ├── 07_hik_0574.png
│   │   │   ├── 08_hik_0566.png
│   │   │   ├── 09_hik_0564.png
│   │   │   └── 10_hik_0556.png
│   │   ├── all_val_detailed_metrics.csv
│   │   ├── summary.json
│   │   ├── worst10_metrics.csv
│   │   └── worst10_overview.png
│   ├── session4_3_smoke_v1/
│   │   ├── config.json
│   │   ├── resnet18_unet_smoke_test.pth
│   │   └── training_log.csv
│   ├── session4_4_evaluation_v1/
│   │   ├── per_image_metrics.csv
│   │   ├── predictions_preview.png
│   │   └── summary.txt
│   ├── session4_4_transfer_v1/
│   │   ├── best_resnet18_unet.pth
│   │   ├── config.json
│   │   ├── history.csv
│   │   └── runtime.txt
│   ├── session4_5_comparison_v1/
│   │   ├── comparison_summary.json
│   │   ├── largest_custom_gains.csv
│   │   ├── largest_transfer_gains.csv
│   │   ├── paired_dice_differences.png
│   │   ├── paired_metrics.csv
│   │   ├── worst_custom.csv
│   │   └── worst_transfer.csv
│   ├── Session5_1_GT_Pilot20_Final_v1/
│   │   ├── masks/
│   │   │   ├── C01L01正.png
│   │   │   ├── C02L01正.png
│   │   │   ├── C03L01正.png
│   │   │   ├── C04L01正.png
│   │   │   ├── C05L01正.png
│   │   │   ├── D3C01L01正.png
│   │   │   ├── D3C02L01正.png
│   │   │   ├── D3C03L01正.png
│   │   │   ├── D3C04L01正.png
│   │   │   ├── D3C05L01正.png
│   │   │   ├── D5C01L01正.png
│   │   │   ├── D5C02L01正.png
│   │   │   ├── D5C03L01正.png
│   │   │   ├── D5C04L01正.png
│   │   │   ├── D5C05L01正.png
│   │   │   ├── D7C01L01正.png
│   │   │   ├── D7C02L01正.png
│   │   │   ├── D7C03L01正.png
│   │   │   ├── D7C04L01正.png
│   │   │   └── D7C05L01正.png
│   │   ├── review_previews/
│   │   │   ├── C01_D0_D3_D5_D7.jpg
│   │   │   ├── C02_D0_D3_D5_D7.jpg
│   │   │   ├── C03_D0_D3_D5_D7.jpg
│   │   │   ├── C04_D0_D3_D5_D7.jpg
│   │   │   └── C05_D0_D3_D5_D7.jpg
│   │   ├── reviewed_json/
│   │   │   ├── C01L01正.json
│   │   │   ├── C02L01正.json
│   │   │   ├── C03L01正.json
│   │   │   ├── C04L01正.json
│   │   │   ├── C05L01正.json
│   │   │   ├── D3C01L01正.json
│   │   │   ├── D3C02L01正.json
│   │   │   ├── D3C03L01正.json
│   │   │   ├── D3C04L01正.json
│   │   │   ├── D3C05L01正.json
│   │   │   ├── D5C01L01正.json
│   │   │   ├── D5C02L01正.json
│   │   │   ├── D5C03L01正.json
│   │   │   ├── D5C04L01正.json
│   │   │   ├── D5C05L01正.json
│   │   │   ├── D7C01L01正.json
│   │   │   ├── D7C02L01正.json
│   │   │   ├── D7C03L01正.json
│   │   │   ├── D7C04L01正.json
│   │   │   └── D7C05L01正.json
│   │   ├── gt_manifest.csv
│   │   └── README.txt
│   ├── session5_1_pilot20_review_v1/
│   │   ├── candidate_json/
│   │   │   ├── C01L01正.json
│   │   │   ├── C02L01正.json
│   │   │   ├── C03L01正.json
│   │   │   ├── C04L01正.json
│   │   │   ├── C05L01正.json
│   │   │   ├── D3C01L01正.json
│   │   │   ├── D3C02L01正.json
│   │   │   ├── D3C03L01正.json
│   │   │   ├── D3C04L01正.json
│   │   │   ├── D3C05L01正.json
│   │   │   ├── D5C01L01正.json
│   │   │   ├── D5C02L01正.json
│   │   │   ├── D5C03L01正.json
│   │   │   ├── D5C04L01正.json
│   │   │   ├── D5C05L01正.json
│   │   │   ├── D7C01L01正.json
│   │   │   ├── D7C02L01正.json
│   │   │   ├── D7C03L01正.json
│   │   │   ├── D7C04L01正.json
│   │   │   └── D7C05L01正.json
│   │   ├── candidate_masks/
│   │   │   ├── C01L01正.png
│   │   │   ├── C02L01正.png
│   │   │   ├── C03L01正.png
│   │   │   ├── C04L01正.png
│   │   │   ├── C05L01正.png
│   │   │   ├── D3C01L01正.png
│   │   │   ├── D3C02L01正.png
│   │   │   ├── D3C03L01正.png
│   │   │   ├── D3C04L01正.png
│   │   │   ├── D3C05L01正.png
│   │   │   ├── D5C01L01正.png
│   │   │   ├── D5C02L01正.png
│   │   │   ├── D5C03L01正.png
│   │   │   ├── D5C04L01正.png
│   │   │   ├── D5C05L01正.png
│   │   │   ├── D7C01L01正.png
│   │   │   ├── D7C02L01正.png
│   │   │   ├── D7C03L01正.png
│   │   │   ├── D7C04L01正.png
│   │   │   └── D7C05L01正.png
│   │   ├── images/
│   │   │   ├── C01L01正.jpg
│   │   │   ├── C01L01正.json
│   │   │   ├── C02L01正.jpg
│   │   │   ├── C02L01正.json
│   │   │   ├── C03L01正.jpg
│   │   │   ├── C03L01正.json
│   │   │   ├── C04L01正.jpg
│   │   │   ├── C04L01正.json
│   │   │   ├── C05L01正.jpg
│   │   │   ├── C05L01正.json
│   │   │   ├── D3C01L01正.jpg
│   │   │   ├── D3C01L01正.json
│   │   │   ├── D3C02L01正.jpg
│   │   │   ├── D3C02L01正.json
│   │   │   ├── D3C03L01正.jpg
│   │   │   ├── D3C03L01正.json
│   │   │   ├── D3C04L01正.jpg
│   │   │   ├── D3C04L01正.json
│   │   │   ├── D3C05L01正.jpg
│   │   │   ├── D3C05L01正.json
│   │   │   ├── D5C01L01正.jpg
│   │   │   ├── D5C01L01正.json
│   │   │   ├── D5C02L01正.jpg
│   │   │   ├── D5C02L01正.json
│   │   │   ├── D5C03L01正.jpg
│   │   │   ├── D5C03L01正.json
│   │   │   ├── D5C04L01正.jpg
│   │   │   ├── D5C04L01正.json
│   │   │   ├── D5C05L01正.jpg
│   │   │   ├── D5C05L01正.json
│   │   │   ├── D7C01L01正.jpg
│   │   │   ├── D7C01L01正.json
│   │   │   ├── D7C02L01正.jpg
│   │   │   ├── D7C02L01正.json
│   │   │   ├── D7C03L01正.jpg
│   │   │   ├── D7C03L01正.json
│   │   │   ├── D7C04L01正.jpg
│   │   │   ├── D7C04L01正.json
│   │   │   ├── D7C05L01正.jpg
│   │   │   └── D7C05L01正.json
│   │   ├── Session5_1_GT_Pilot20_Final_v1/
│   │   │   ├── masks/
│   │   │   │   ├── C01L01正.png
│   │   │   │   ├── C02L01正.png
│   │   │   │   ├── C03L01正.png
│   │   │   │   ├── C04L01正.png
│   │   │   │   ├── C05L01正.png
│   │   │   │   ├── D3C01L01正.png
│   │   │   │   ├── D3C02L01正.png
│   │   │   │   ├── D3C03L01正.png
│   │   │   │   ├── D3C04L01正.png
│   │   │   │   ├── D3C05L01正.png
│   │   │   │   ├── D5C01L01正.png
│   │   │   │   ├── D5C02L01正.png
│   │   │   │   ├── D5C03L01正.png
│   │   │   │   ├── D5C04L01正.png
│   │   │   │   ├── D5C05L01正.png
│   │   │   │   ├── D7C01L01正.png
│   │   │   │   ├── D7C02L01正.png
│   │   │   │   ├── D7C03L01正.png
│   │   │   │   ├── D7C04L01正.png
│   │   │   │   └── D7C05L01正.png
│   │   │   ├── review_previews/
│   │   │   │   ├── C01_D0_D3_D5_D7.jpg
│   │   │   │   ├── C02_D0_D3_D5_D7.jpg
│   │   │   │   ├── C03_D0_D3_D5_D7.jpg
│   │   │   │   ├── C04_D0_D3_D5_D7.jpg
│   │   │   │   └── C05_D0_D3_D5_D7.jpg
│   │   │   ├── reviewed_json/
│   │   │   │   ├── C01L01正.json
│   │   │   │   ├── C02L01正.json
│   │   │   │   ├── C03L01正.json
│   │   │   │   ├── C04L01正.json
│   │   │   │   ├── C05L01正.json
│   │   │   │   ├── D3C01L01正.json
│   │   │   │   ├── D3C02L01正.json
│   │   │   │   ├── D3C03L01正.json
│   │   │   │   ├── D3C04L01正.json
│   │   │   │   ├── D3C05L01正.json
│   │   │   │   ├── D5C01L01正.json
│   │   │   │   ├── D5C02L01正.json
│   │   │   │   ├── D5C03L01正.json
│   │   │   │   ├── D5C04L01正.json
│   │   │   │   ├── D5C05L01正.json
│   │   │   │   ├── D7C01L01正.json
│   │   │   │   ├── D7C02L01正.json
│   │   │   │   ├── D7C03L01正.json
│   │   │   │   ├── D7C04L01正.json
│   │   │   │   └── D7C05L01正.json
│   │   │   ├── gt_manifest.csv
│   │   │   └── README.txt
│   │   ├── Pilot20_C01_review.jpg
│   │   ├── Pilot20_C02_review.jpg
│   │   ├── Pilot20_C03_review.jpg
│   │   ├── Pilot20_C04_review.jpg
│   │   ├── Pilot20_C05_review.jpg
│   │   ├── README.txt
│   │   ├── review_tracking.csv
│   │   ├── Session5_Pilot20_QC_Report.csv
│   │   └── Session5_Pilot20_Reviewed.zip
│   ├── session5_1_target_annotation_v1/
│   │   ├── images/
│   │   │   ├── D0C01L01正.jpg
│   │   │   ├── D0C01L01正.json
│   │   │   ├── D0C02L01正.jpg
│   │   │   ├── D0C02L01正.json
│   │   │   ├── D0C03L01正.jpg
│   │   │   ├── D0C03L01正.json
│   │   │   ├── D0C04L01正.jpg
│   │   │   ├── D0C04L01正.json
│   │   │   ├── D0C05L01正.jpg
│   │   │   ├── D0C05L01正.json
│   │   │   ├── D3C01L01正.jpg
│   │   │   ├── D3C01L01正.json
│   │   │   ├── D3C02L01正.jpg
│   │   │   ├── D3C02L01正.json
│   │   │   ├── D3C03L01正.jpg
│   │   │   ├── D3C04L01正.jpg
│   │   │   ├── D3C05L01正.jpg
│   │   │   ├── D5C01L01正.jpg
│   │   │   ├── D5C02L01正.jpg
│   │   │   ├── D5C03L01正.jpg
│   │   │   ├── D5C04L01正.jpg
│   │   │   ├── D5C05L01正.jpg
│   │   │   ├── D7C01L01正.jpg
│   │   │   ├── D7C01L01正.json
│   │   │   ├── D7C02L01正.jpg
│   │   │   ├── D7C03L01正.jpg
│   │   │   ├── D7C04L01正.jpg
│   │   │   └── D7C05L01正.jpg
│   │   ├── annotation_manifest.csv
│   │   └── ANNOTATION_POLICY.txt
│   ├── session5_2_1_diagnostics_v1/
│   │   ├── diagnosis_summary.json
│   │   ├── source_per_image_diagnostic.csv
│   │   ├── target_per_image_diagnostic.csv
│   │   ├── target_probability_diagnostic.csv
│   │   └── threshold_sweep_DIAGNOSTIC_ONLY.csv
│   ├── session5_2_zero_shot_v1/
│   │   ├── previews/
│   │   │   ├── C01_RGB_GT_two_models.jpg
│   │   │   ├── C02_RGB_GT_two_models.jpg
│   │   │   ├── C03_RGB_GT_two_models.jpg
│   │   │   ├── C04_RGB_GT_two_models.jpg
│   │   │   └── C05_RGB_GT_two_models.jpg
│   │   ├── group_metrics.csv
│   │   ├── per_image_metrics.csv
│   │   └── summary.json
│   ├── session5_3_custom_loho_v1/
│   │   ├── fold_membership.csv
│   │   ├── oof_group_metrics.csv
│   │   ├── oof_per_image_metrics.csv
│   │   ├── protocol_summary.json
│   │   └── training_history.csv
│   ├── session5_3_model_comparison_v1/
│   │   ├── model_summary.csv
│   │   ├── paired_oof_20_images.csv
│   │   ├── per_day_mean_dice.csv
│   │   ├── per_head_mean_dice.csv
│   │   └── README.txt
│   ├── session5_3_resnet18_loho_v1/
│   │   ├── fold_membership.csv
│   │   ├── oof_group_metrics.csv
│   │   ├── oof_per_image_metrics.csv
│   │   ├── protocol_summary.json
│   │   └── training_history.csv
│   ├── session6_1_longitudinal_audit_v1/
│   │   ├── audit_summary.json
│   │   ├── D0_followup_pairs_111.csv
│   │   ├── expected_grid_160.csv
│   │   ├── longitudinal_series_40.csv
│   │   ├── missing_observations.csv
│   │   ├── observation_manifest_157.csv
│   │   └── README_RESEARCH_BOUNDARIES.txt
│   ├── Session6_1_Longitudinal_Audit_v1---1/
│   │   ├── audit_summary.json
│   │   ├── D0_followup_pairs_111.csv
│   │   ├── expected_grid_160.csv
│   │   ├── longitudinal_series_40.csv
│   │   ├── missing_observations.csv
│   │   ├── observation_manifest_157.csv
│   │   └── README_RESEARCH_BOUNDARIES.txt
│   ├── session6_2_analysis_roi_candidates_v1/
│   │   ├── masks_128/
│   │   │   ├── D0C01L01反.png
│   │   │   ├── D0C01L01正.png
│   │   │   ├── D0C01L02反.png
│   │   │   ├── D0C01L02正.png
│   │   │   ├── D0C01L03反.png
│   │   │   ├── D0C01L03正.png
│   │   │   ├── D0C01L04反.png
│   │   │   ├── D0C01L04正.png
│   │   │   ├── D0C02L01反.png
│   │   │   ├── D0C02L01正.png
│   │   │   ├── D0C02L02反.png
│   │   │   ├── D0C02L02正.png
│   │   │   ├── D0C02L03反.png
│   │   │   ├── D0C02L03正.png
│   │   │   ├── D0C02L04反.png
│   │   │   ├── D0C02L04正.png
│   │   │   ├── D0C03L01反.png
│   │   │   ├── D0C03L01正.png
│   │   │   ├── D0C03L02反.png
│   │   │   ├── D0C03L02正.png
│   │   │   ├── D0C03L03反.png
│   │   │   ├── D0C03L03正.png
│   │   │   ├── D0C03L04反.png
│   │   │   ├── D0C03L04正.png
│   │   │   ├── D0C04L01反.png
│   │   │   ├── D0C04L01正.png
│   │   │   ├── D0C04L02反.png
│   │   │   ├── D0C04L02正.png
│   │   │   ├── D0C04L03正.png
│   │   │   ├── D0C04L04反.png
│   │   │   ├── D0C04L04正.png
│   │   │   ├── D0C05L01反.png
│   │   │   ├── D0C05L01正.png
│   │   │   ├── D0C05L02反.png
│   │   │   ├── D0C05L02正.png
│   │   │   ├── D0C05L03正.png
│   │   │   ├── D0C05L04正.png
│   │   │   ├── D3C01L01反.png
│   │   │   ├── D3C01L01正.png
│   │   │   ├── D3C01L02反.png
│   │   │   ├── D3C01L02正.png
│   │   │   ├── D3C01L03反.png
│   │   │   ├── D3C01L03正.png
│   │   │   ├── D3C01L04反.png
│   │   │   ├── D3C01L04正.png
│   │   │   ├── D3C02L01反.png
│   │   │   ├── D3C02L01正.png
│   │   │   ├── D3C02L02反.png
│   │   │   ├── D3C02L02正.png
│   │   │   ├── D3C02L03反.png
│   │   │   ├── D3C02L03正.png
│   │   │   ├── D3C02L04反.png
│   │   │   ├── D3C02L04正.png
│   │   │   ├── D3C03L01反.png
│   │   │   ├── D3C03L01正.png
│   │   │   ├── D3C03L02反.png
│   │   │   ├── D3C03L02正.png
│   │   │   ├── D3C03L03反.png
│   │   │   ├── D3C03L03正.png
│   │   │   ├── D3C03L04反.png
│   │   │   ├── D3C03L04正.png
│   │   │   ├── D3C04L01反.png
│   │   │   ├── D3C04L01正.png
│   │   │   ├── D3C04L02反.png
│   │   │   ├── D3C04L02正.png
│   │   │   ├── D3C04L03反.png
│   │   │   ├── D3C04L03正.png
│   │   │   ├── D3C04L04反.png
│   │   │   ├── D3C04L04正.png
│   │   │   ├── D3C05L01反.png
│   │   │   ├── D3C05L01正.png
│   │   │   ├── D3C05L02反.png
│   │   │   ├── D3C05L02正.png
│   │   │   ├── D3C05L03反.png
│   │   │   ├── D3C05L03正.png
│   │   │   ├── D3C05L04反.png
│   │   │   ├── D3C05L04正.png
│   │   │   ├── D5C01L01反.png
│   │   │   ├── D5C01L01正.png
│   │   │   ├── D5C01L02反.png
│   │   │   ├── D5C01L02正.png
│   │   │   ├── D5C01L03反.png
│   │   │   ├── D5C01L03正.png
│   │   │   ├── D5C01L04反.png
│   │   │   ├── D5C01L04正.png
│   │   │   ├── D5C02L01反.png
│   │   │   ├── D5C02L01正.png
│   │   │   ├── D5C02L02反.png
│   │   │   ├── D5C02L02正.png
│   │   │   ├── D5C02L03反.png
│   │   │   ├── D5C02L03正.png
│   │   │   ├── D5C02L04反.png
│   │   │   ├── D5C02L04正.png
│   │   │   ├── D5C03L01反.png
│   │   │   ├── D5C03L01正.png
│   │   │   ├── D5C03L02反.png
│   │   │   ├── D5C03L02正.png
│   │   │   ├── D5C03L03反.png
│   │   │   ├── D5C03L03正.png
│   │   │   ├── D5C03L04反.png
│   │   │   ├── D5C03L04正.png
│   │   │   ├── D5C04L01反.png
│   │   │   ├── D5C04L01正.png
│   │   │   ├── D5C04L02反.png
│   │   │   ├── D5C04L02正.png
│   │   │   ├── D5C04L03反.png
│   │   │   ├── D5C04L03正.png
│   │   │   ├── D5C04L04反.png
│   │   │   ├── D5C04L04正.png
│   │   │   ├── D5C05L01反.png
│   │   │   ├── D5C05L01正.png
│   │   │   ├── D5C05L02反.png
│   │   │   ├── D5C05L02正.png
│   │   │   ├── D5C05L03反.png
│   │   │   ├── D5C05L03正.png
│   │   │   ├── D5C05L04反.png
│   │   │   ├── D5C05L04正.png
│   │   │   ├── D7C01L01反.png
│   │   │   ├── D7C01L01正.png
│   │   │   ├── D7C01L02反.png
│   │   │   ├── D7C01L02正.png
│   │   │   ├── D7C01L03反.png
│   │   │   ├── D7C01L03正.png
│   │   │   ├── D7C01L04反.png
│   │   │   ├── D7C01L04正.png
│   │   │   ├── D7C02L01反.png
│   │   │   ├── D7C02L01正.png
│   │   │   ├── D7C02L02反.png
│   │   │   ├── D7C02L02正.png
│   │   │   ├── D7C02L03反.png
│   │   │   ├── D7C02L03正.png
│   │   │   ├── D7C02L04反.png
│   │   │   ├── D7C02L04正.png
│   │   │   ├── D7C03L01反.png
│   │   │   ├── D7C03L01正.png
│   │   │   ├── D7C03L02反.png
│   │   │   ├── D7C03L02正.png
│   │   │   ├── D7C03L03反.png
│   │   │   ├── D7C03L03正.png
│   │   │   ├── D7C03L04反.png
│   │   │   ├── D7C03L04正.png
│   │   │   ├── D7C04L01反.png
│   │   │   ├── D7C04L01正.png
│   │   │   ├── D7C04L02反.png
│   │   │   ├── D7C04L02正.png
│   │   │   ├── D7C04L03反.png
│   │   │   ├── D7C04L03正.png
│   │   │   ├── D7C04L04反.png
│   │   │   ├── D7C04L04正.png
│   │   │   ├── D7C05L01反.png
│   │   │   ├── D7C05L01正.png
│   │   │   ├── D7C05L02反.png
│   │   │   ├── D7C05L02正.png
│   │   │   ├── D7C05L03反.png
│   │   │   ├── D7C05L03正.png
│   │   │   ├── D7C05L04反.png
│   │   │   └── D7C05L04正.png
│   │   ├── masks_full_resolution/
│   │   │   ├── D0C01L01反.png
│   │   │   ├── D0C01L01正.png
│   │   │   ├── D0C01L02反.png
│   │   │   ├── D0C01L02正.png
│   │   │   ├── D0C01L03反.png
│   │   │   ├── D0C01L03正.png
│   │   │   ├── D0C01L04反.png
│   │   │   ├── D0C01L04正.png
│   │   │   ├── D0C02L01反.png
│   │   │   ├── D0C02L01正.png
│   │   │   ├── D0C02L02反.png
│   │   │   ├── D0C02L02正.png
│   │   │   ├── D0C02L03反.png
│   │   │   ├── D0C02L03正.png
│   │   │   ├── D0C02L04反.png
│   │   │   ├── D0C02L04正.png
│   │   │   ├── D0C03L01反.png
│   │   │   ├── D0C03L01正.png
│   │   │   ├── D0C03L02反.png
│   │   │   ├── D0C03L02正.png
│   │   │   ├── D0C03L03反.png
│   │   │   ├── D0C03L03正.png
│   │   │   ├── D0C03L04反.png
│   │   │   ├── D0C03L04正.png
│   │   │   ├── D0C04L01反.png
│   │   │   ├── D0C04L01正.png
│   │   │   ├── D0C04L02反.png
│   │   │   ├── D0C04L02正.png
│   │   │   ├── D0C04L03正.png
│   │   │   ├── D0C04L04反.png
│   │   │   ├── D0C04L04正.png
│   │   │   ├── D0C05L01反.png
│   │   │   ├── D0C05L01正.png
│   │   │   ├── D0C05L02反.png
│   │   │   ├── D0C05L02正.png
│   │   │   ├── D0C05L03正.png
│   │   │   ├── D0C05L04正.png
│   │   │   ├── D3C01L01反.png
│   │   │   ├── D3C01L01正.png
│   │   │   ├── D3C01L02反.png
│   │   │   ├── D3C01L02正.png
│   │   │   ├── D3C01L03反.png
│   │   │   ├── D3C01L03正.png
│   │   │   ├── D3C01L04反.png
│   │   │   ├── D3C01L04正.png
│   │   │   ├── D3C02L01反.png
│   │   │   ├── D3C02L01正.png
│   │   │   ├── D3C02L02反.png
│   │   │   ├── D3C02L02正.png
│   │   │   ├── D3C02L03反.png
│   │   │   ├── D3C02L03正.png
│   │   │   ├── D3C02L04反.png
│   │   │   ├── D3C02L04正.png
│   │   │   ├── D3C03L01反.png
│   │   │   ├── D3C03L01正.png
│   │   │   ├── D3C03L02反.png
│   │   │   ├── D3C03L02正.png
│   │   │   ├── D3C03L03反.png
│   │   │   ├── D3C03L03正.png
│   │   │   ├── D3C03L04反.png
│   │   │   ├── D3C03L04正.png
│   │   │   ├── D3C04L01反.png
│   │   │   ├── D3C04L01正.png
│   │   │   ├── D3C04L02反.png
│   │   │   ├── D3C04L02正.png
│   │   │   ├── D3C04L03反.png
│   │   │   ├── D3C04L03正.png
│   │   │   ├── D3C04L04反.png
│   │   │   ├── D3C04L04正.png
│   │   │   ├── D3C05L01反.png
│   │   │   ├── D3C05L01正.png
│   │   │   ├── D3C05L02反.png
│   │   │   ├── D3C05L02正.png
│   │   │   ├── D3C05L03反.png
│   │   │   ├── D3C05L03正.png
│   │   │   ├── D3C05L04反.png
│   │   │   ├── D3C05L04正.png
│   │   │   ├── D5C01L01反.png
│   │   │   ├── D5C01L01正.png
│   │   │   ├── D5C01L02反.png
│   │   │   ├── D5C01L02正.png
│   │   │   ├── D5C01L03反.png
│   │   │   ├── D5C01L03正.png
│   │   │   ├── D5C01L04反.png
│   │   │   ├── D5C01L04正.png
│   │   │   ├── D5C02L01反.png
│   │   │   ├── D5C02L01正.png
│   │   │   ├── D5C02L02反.png
│   │   │   ├── D5C02L02正.png
│   │   │   ├── D5C02L03反.png
│   │   │   ├── D5C02L03正.png
│   │   │   ├── D5C02L04反.png
│   │   │   ├── D5C02L04正.png
│   │   │   ├── D5C03L01反.png
│   │   │   ├── D5C03L01正.png
│   │   │   ├── D5C03L02反.png
│   │   │   ├── D5C03L02正.png
│   │   │   ├── D5C03L03反.png
│   │   │   ├── D5C03L03正.png
│   │   │   ├── D5C03L04反.png
│   │   │   ├── D5C03L04正.png
│   │   │   ├── D5C04L01反.png
│   │   │   ├── D5C04L01正.png
│   │   │   ├── D5C04L02反.png
│   │   │   ├── D5C04L02正.png
│   │   │   ├── D5C04L03反.png
│   │   │   ├── D5C04L03正.png
│   │   │   ├── D5C04L04反.png
│   │   │   ├── D5C04L04正.png
│   │   │   ├── D5C05L01反.png
│   │   │   ├── D5C05L01正.png
│   │   │   ├── D5C05L02反.png
│   │   │   ├── D5C05L02正.png
│   │   │   ├── D5C05L03反.png
│   │   │   ├── D5C05L03正.png
│   │   │   ├── D5C05L04反.png
│   │   │   ├── D5C05L04正.png
│   │   │   ├── D7C01L01反.png
│   │   │   ├── D7C01L01正.png
│   │   │   ├── D7C01L02反.png
│   │   │   ├── D7C01L02正.png
│   │   │   ├── D7C01L03反.png
│   │   │   ├── D7C01L03正.png
│   │   │   ├── D7C01L04反.png
│   │   │   ├── D7C01L04正.png
│   │   │   ├── D7C02L01反.png
│   │   │   ├── D7C02L01正.png
│   │   │   ├── D7C02L02反.png
│   │   │   ├── D7C02L02正.png
│   │   │   ├── D7C02L03反.png
│   │   │   ├── D7C02L03正.png
│   │   │   ├── D7C02L04反.png
│   │   │   ├── D7C02L04正.png
│   │   │   ├── D7C03L01反.png
│   │   │   ├── D7C03L01正.png
│   │   │   ├── D7C03L02反.png
│   │   │   ├── D7C03L02正.png
│   │   │   ├── D7C03L03反.png
│   │   │   ├── D7C03L03正.png
│   │   │   ├── D7C03L04反.png
│   │   │   ├── D7C03L04正.png
│   │   │   ├── D7C04L01反.png
│   │   │   ├── D7C04L01正.png
│   │   │   ├── D7C04L02反.png
│   │   │   ├── D7C04L02正.png
│   │   │   ├── D7C04L03反.png
│   │   │   ├── D7C04L03正.png
│   │   │   ├── D7C04L04反.png
│   │   │   ├── D7C04L04正.png
│   │   │   ├── D7C05L01反.png
│   │   │   ├── D7C05L01正.png
│   │   │   ├── D7C05L02反.png
│   │   │   ├── D7C05L02正.png
│   │   │   ├── D7C05L03反.png
│   │   │   ├── D7C05L03正.png
│   │   │   ├── D7C05L04反.png
│   │   │   └── D7C05L04正.png
│   │   ├── previews_by_day/
│   │   │   ├── D0_all_candidates.jpg
│   │   │   ├── D3_all_candidates.jpg
│   │   │   ├── D5_all_candidates.jpg
│   │   │   └── D7_all_candidates.jpg
│   │   ├── analysis_model_protocol.json
│   │   ├── candidate_roi_manifest_157.csv
│   │   ├── resnet18_target_analysis_only.pth
│   │   └── training_history.csv
│   ├── session6_2_roi_qc_review_v1/
│   │   ├── D0_all_candidates.jpg
│   │   ├── D3_all_candidates.jpg
│   │   ├── D5_all_candidates.jpg
│   │   ├── D7_all_candidates.jpg
│   │   ├── README_QC.txt
│   │   ├── ROI_QC_157_review_tracking.csv
│   │   └── ROI_QC_summary.json
│   ├── session6_3_provisional_review_v1/
│   │   ├── delta_brown_candidate_ratio_by_head.png
│   │   ├── delta_green_ratio_by_head.png
│   │   ├── delta_roi_fraction_by_head.png
│   │   ├── delta_yellow_candidate_ratio_by_head.png
│   │   └── README_Analysis.txt
│   ├── session6_3_visual_features_provisional_v1/
│   │   ├── analysis_protocol.json
│   │   ├── D0_followup_deltas_111_PROVISIONAL.csv
│   │   ├── descriptive_by_day_side_PROVISIONAL.csv
│   │   └── visual_features_157_PROVISIONAL.csv
│   ├── session6_4_2_photo_color_diagnostics_v1/
│   │   ├── by_day_5heads.csv
│   │   ├── paired_D0_changes_15.csv
│   │   ├── pilot20_leaf_wood_color.csv
│   │   ├── protocol_and_limitations.json
│   │   ├── README_RESULTS.txt
│   │   └── threshold_consistency_5heads.csv
│   ├── session6_4_2_photo_color_reproduction_v1/
│   │   ├── by_day_5heads.csv
│   │   ├── paired_D0_changes_15.csv
│   │   ├── pilot20_leaf_wood_color.csv
│   │   ├── protocol_and_limitations.json
│   │   └── threshold_consistency_5heads.csv
│   ├── session6_4_gt_auto_sensitivity_v2/
│   │   ├── by_day_GT_vs_Auto.csv
│   │   ├── diagnosis_summary.json
│   │   ├── paired_D0_changes_GT_vs_Auto_15.csv
│   │   └── per_image_GT_vs_Auto_20.csv
│   ├── session6_4_gt_pilot20_reference_v1/
│   │   ├── GT_pilot20_day_means.csv
│   │   ├── GT_pilot20_features_reference.csv
│   │   └── README.txt
│   ├── session6_5_report_results_v1/
│   │   ├── Evidence_and_limitations.md
│   │   ├── Figure1_segmentation_comparison.png
│   │   ├── Figure2_GT_vs_auto_yellow.png
│   │   ├── Figure3_HSV_threshold_sensitivity.png
│   │   ├── Figure4_wood_reference_RGB.png
│   │   ├── Figure5_projected_fraction.png
│   │   ├── README.txt
│   │   ├── Report_results_simple_English.md
│   │   ├── Table1_segmentation_comparison.csv
│   │   ├── Table2_GT_vs_auto_by_day.csv
│   │   ├── Table3_threshold_and_wood_reference.csv
│   │   ├── Table4_threshold_direction_consistency.csv
│   │   └── Table5_full157_provisional_by_day_side.csv
│   ├── E3_Session5_1_Pilot20_GT_Final_v1.zip
│   └── session6_4_gt_auto_sensitivity_v2.zip
├── submission_preparation/
│   ├── audit_results_v2/
│   │   ├── artifact_audit_summary_v2.json
│   │   ├── artifact_audit_summary_v2.md
│   │   └── artifact_inventory_v2.csv
│   ├── audit_model_artifact_v2.py
│   └── README_Artifact_Audit_v2.txt
├── demo.py
├── Readme.md
└── session3_1_pair_preview.png
```

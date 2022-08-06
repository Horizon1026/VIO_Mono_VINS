# VIO-Mono
A simple vio-mono with two backend solver (scba and rootba).

# 作者备注
+ 1、这是一个简单的单目VIO，内置了舒尔补和平方根两种后端优化求解器。
+ 2、此VIO本质上是对港科大VINS-Mono的重构和简化，注释比较详细，适合VINS入门学习用。
+ 3、此VIO在编写时已经脱离ros环境，仅需适配“图像输入”、“IMU输入”、“位姿估计输出”即可运行，工程源码中的ros部分仅作接口适配。
+ 4、此VIO的平方根后端优化求解器还不是很稳定，若有兴趣可以一起交流学习改进。
+ 5、源码仅供交流学习使用，暂不同意商用。

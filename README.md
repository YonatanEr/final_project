
Polarnet on Ouster-Os1 128 driving:
![image](https://github.com/YonatanEr/final_project/assets/43711624/3f7ab29f-9d22-445a-9087-6c0de6c086fd)
standby : 
![image](https://github.com/YonatanEr/final_project/assets/43711624/520b5fa7-d4f9-47a2-8f67-6517b1784826)
To download the pretrained weigths go to https://github.com/edwardzhou130/PolarSeg then added them to the pretrained weigths dir in this git repo. 
To interface with Lidar we used - https://github.com/ouster-lidar/ouster-ros?tab=readme-ov-file
To build the Lidar network using ROS you need to clone the NN_node directory into the src folder of ros2_ws as shown below : 
cd ros2_ws && mkdir -p src && git clone https://github.com/YonatanEr/final_project.git src/NN_node
then go to the ros2_ws directory again and build the new via  :colcon build --packages-select nn_node




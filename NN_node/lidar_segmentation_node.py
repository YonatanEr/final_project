#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import argparse
import sys
import numpy as np
import time 
import torch
import torch.optim as optim
from tqdm import tqdm
from std_msgs.msg import Int32MultiArray
import yaml
from network.BEV_Unet import BEV_Unet
from network.ptBEV import ptBEVnet
from dataloader.dataset import collate_fn_BEV,collate_fn_BEV_test,SemKITTI,SemKITTI_label_name,spherical_dataset,voxel_dataset
#ignore weird np warning
import warnings
from std_msgs.msg import Header
warnings.filterwarnings("ignore")
import subprocess
from sensor_msgs.msg import PointCloud2
import struct
import time
from builtin_interfaces.msg import Time
class LidarSegmentationNode(Node):
    def __init__(self, debug_mode=False, output_directory=None, label_output_directory=None):
        super().__init__('lidar_segmentation_node')
        #listen to lidar data
        self.subscription = self.create_subscription(PointCloud2,'/ouster/points',self.listener_callback,10)
        #published network output of labeled data
        self.label_publisher = self.create_publisher(Int32MultiArray, '/lidar/processed_labels', 10)
        self.data_publisher = self.create_publisher(Float32MultiArray,'/lidar/raw_data',10)
        self.timestamp_publisher = self.create_publisher(Time, '/lidar/timestamp', 10)
        compression_model=32
        #need to first process unique labels
        unique_label=np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
        unique_label_str=[SemKITTI_label_name[x] for x in unique_label+1]
        # Load the model
        self.model = self.load_model("polar", unique_label, compression_model, [480, 360, 32],"/home/eeproj1/ros2_ws/src/NN_node/pretrained_weight/SemKITTI_PolarSeg.pt", torch.device('cuda:0'))
        ####this is for labeled remapping to later use#######
        self.remap_config = self.load_remap_config('/home/eeproj1/semantic-api/semantic-kitti-api/config/semantic-kitti.yaml')
        self.remap_lut = self.create_remap_lut(self.remap_config, inverse=True)
        #####################################################
        self.count=0# a counter to keep track
        self.debug_mode = debug_mode
        self.output_directory = output_directory
        self.label_output_directory = label_output_directory
        self.counter = 0

     
    def process_point_cloud(self, msg):
        print("lidar data received processing......")
        point_data = Float32MultiArray()
        data=[]
        for i in range(0, len(msg.data), msg.point_step):  
          x, = struct.unpack_from('f', msg.data, i + 0)  
          y, = struct.unpack_from('f', msg.data, i + 4)  
          z, = struct.unpack_from('f', msg.data, i + 8)  
          intensity, = struct.unpack_from('H', msg.data, i + 24)
          data.append(x)
          data.append(y)
          data.append(z)
          data.append(intensity)
          point_data.data.extend([x, y, z, intensity])
         
        return point_data,data
        
    def train2SemKITTI(self,input_label):
        # delete 0 label (uses uint8 trick : 0 - 1 = 255 )
        return input_label + 1
        
    def load_remap_config(self, config_path):
        return yaml.safe_load(open(config_path, 'r'))
    
    def create_remap_lut(self, config, inverse=False):
        remap_dict = config['learning_map_inv'] if inverse else config['learning_map']
        max_key = max(remap_dict.keys())
        remap_lut = np.zeros((max_key + 100), dtype=np.int32)
        remap_lut[list(remap_dict.keys())] = list(remap_dict.values())
        return remap_lut

    def load_model(self,model_type, unique_label, compression_model, grid_size, model_save_path, pytorch_device):
         # Model type selection: 'polar' or 'traditional'
        if model_type == 'polar':
            fea_dim = 9
            circular_padding = True

         # Model initialization
        my_BEV_model = BEV_Unet(n_class=len(unique_label), n_height=compression_model, input_batch_norm=True, dropout=0.0, circular_padding=circular_padding)
        my_model = ptBEVnet(my_BEV_model, pt_model='pointnet', grid_size=grid_size, fea_dim=fea_dim, max_pt_per_encode=256, out_pt_fea_dim=512, kernal_size=1, pt_selection='random', fea_compre=compression_model)

         # Load pre-trained model weights
        if os.path.exists(model_save_path):
            my_model.load_state_dict(torch.load(model_save_path))
        my_model.to(pytorch_device)  # Moving the model to the GPU

        return my_model

     
    def process_data(self,data_scans,grid_size,test_batch_size):
        # prepare dataset
        test_pt_dataset = SemKITTI(data_scans, imageset = 'test', return_ref = True)
        test_dataset=spherical_dataset(test_pt_dataset, grid_size = grid_size, ignore_label = 0, fixed_volume_space = True, return_test= True)
  
        test_dataset_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                                    batch_size = test_batch_size,
                                                    collate_fn = collate_fn_BEV_test,
                                                    shuffle = False,
                                                    num_workers = 4)
        return  test_dataset_loader, test_pt_dataset     
    
    def save_and_publish_data(self, label_data, point_data):
        # Save label data
        label_filename = f"{str(self.counter).zfill(6)}.label"
        label_filepath = os.path.join(self.label_output_directory, label_filename)
        with open(label_filepath, 'wb') as label_file:
            label_file.write(bytearray(struct.pack('i' * len(label_data), *label_data)))

           # Save point data
        point_filename = f"{str(self.counter).zfill(6)}.bin"
        point_filepath = os.path.join(self.output_directory, point_filename)
        
        with open(point_filepath, 'wb') as point_file:
            point_file.write(bytearray(struct.pack('f' * len(point_data), *point_data)))
        self.counter += 1
        
     
    def publish_data_with_timestamp(self, label_msg, data_scans):
    # Publishing the label and point cloud data as before
      self.label_publisher.publish(label_msg)
      self.data_publisher.publish(data_scans)
      #print(label_msg.data)
      #print(data_scans.data)
    # Get current time and publish
      now = self.get_clock().now()
      time_msg = Time()
      time_msg.sec = now.seconds_nanoseconds()[0]
      time_msg.nanosec = now.seconds_nanoseconds()[1]
      self.timestamp_publisher.publish(time_msg)

      #print(f"Data and timestamp published at {now.to_msg()}")
        
    def listener_callback(self, msg):
         #process the point cloud data to someting the model can get as input
         data_scans,data = self.process_point_cloud(msg)
       
         print("start predicting labels")
         test_dataset_loader, test_pt_dataset =self.process_data(data,[480, 360, 32],1)#prepare data for model.... converting raw data to grid
          # predictions
         pbar = tqdm(total=len(test_dataset_loader))
         with torch.no_grad():
             for i_iter_test,(_,_,test_grid,_,test_pt_fea,test_index) in enumerate(test_dataset_loader):
              # predict
                 test_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(torch.device('cuda:0')) for i in test_pt_fea]
                 test_grid_ten = [torch.from_numpy(i[:,:2]).to(torch.device('cuda:0')) for i in test_grid]
                 torch.cuda.synchronize()  # Wait for GPU operations to complete
                 start_time = time.time()
                 predict_labels = self.model(test_pt_fea_ten,test_grid_ten)
                 predict_labels = torch.argmax(predict_labels,1).type(torch.uint8)#predict grid= here is [[16,17,18............19]],tensor each number is class for example 16=road 18=person
                 torch.cuda.synchronize()  # Wait for GPU operations to complete
                 predict_labels = predict_labels.cpu().detach().numpy()#detached from gpu...... 200 ms hardware
                 # write to label file
                 for count, i_test_grid in enumerate(test_grid):
                   # Get all the labeled data in an organized way, one-to-one map to points
                   test_pred_label = predict_labels[count, test_grid[count][:, 0], test_grid[count][:, 1], test_grid[count][:, 2]]
                   test_pred_label = self.train2SemKITTI(test_pred_label)  # Cleaning
                   test_pred_label = test_pred_label.astype(np.uint32)
    
                   # Publish the labels directly after converting them to uint32
                   label_msg = Int32MultiArray()
                   label_msg.data = test_pred_label.tolist()  # Convert the numpy array to a list for publishing
                   self.publish_data_with_timestamp(label_msg, data_scans)
                 
                  
                   cpu_time = time.time() - start_time
                   print(f"Published time for batch {i_iter_test}: {cpu_time} seconds")
    
                   # SemanticKITTI label manipulation for saving, if debug_mode is enabled
                   if self.debug_mode:
                   # Copy the labels for manipulation
                     manipulated_label = np.copy(test_pred_label)
        
                     # SemanticKITTI-specific manipulation
                     upper_half = manipulated_label >> 16  # Get upper half for instance
                     lower_half = manipulated_label & 0xFFFF  # Get lower half for semantics
                     lower_half = self.remap_lut[lower_half]  # Do the remapping of semantics
                     manipulated_label = (upper_half << 16) + lower_half  # Reconstruct full label
                     manipulated_label = manipulated_label.astype(np.uint32)
        
                     # Now save the manipulated labels instead of publishing them
                     self.save_and_publish_data(manipulated_label, data)
    
                   print(f"Published label data for grid {self.count}")
                   self.count += 1
                   pbar.update(1)
         del test_grid,test_pt_fea,test_index
       
         pbar.close()
         
    

def main(args=None):
    parser = argparse.ArgumentParser(description='LiDAR Segmentation Node')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode to save data')
    parser.add_argument('--output_directory', type=str, default='/path/to/output', help='Output directory for point cloud data')
    parser.add_argument('--label_output_directory', type=str, default='/path/to/labels', help='Output directory for segmented labels')
    args = parser.parse_known_args()[0]

    # Convert Namespace to a list of arguments
    args_list = ['--debug'] if args.debug else []
    if args.output_directory:
        args_list.extend(['--output_directory', args.output_directory])
    if args.label_output_directory:
        args_list.extend(['--label_output_directory', args.label_output_directory])

    rclpy.init(args=args_list)
    segmentation_node = LidarSegmentationNode(debug_mode=args.debug, output_directory=args.output_directory, label_output_directory=args.label_output_directory)
    rclpy.spin(segmentation_node)
    segmentation_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


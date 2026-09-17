#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于YOLO11的CAD图纸检测系统 - PySide6图形界面
原始博客：https://cv2023.blog.csdn.net/article/details/149266015
"""

import sys
import os
import cv2
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                               QTextEdit, QProgressBar, QTabWidget, QGroupBox,
                               QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QPixmap, QImage, QFont
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

class DetectionThread(QThread):
    """检测线程，避免界面卡顿"""
    detection_complete = Signal(object, str)
    progress_update = Signal(int)
    
    def __init__(self, model_path, image_path, conf_threshold=0.5):
        super().__init__()
        self.model_path = model_path
        self.image_path = image_path
        self.conf_threshold = conf_threshold
        
    def run(self):
        """执行检测"""
        try:
            self.progress_update.emit(10)
            
            # 加载模型
            model = YOLO(self.model_path)
            self.progress_update.emit(30)
            
            # 执行检测
            results = model(self.image_path, conf=self.conf_threshold)
            self.progress_update.emit(80)
            
            # 处理结果
            result_image = results[0].plot()
            self.progress_update.emit(90)
            
            # 生成检测结果文本
            detections = results[0].boxes
            result_text = f"检测到 {len(detections)} 个对象:\n"
            
            if len(detections) > 0:
                for box in detections:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = model.names[cls_id]
                    result_text += f"- {class_name}: {conf:.2f}\n"
            else:
                result_text += "未检测到任何对象"
                
            self.progress_update.emit(100)
            self.detection_complete.emit(result_image, result_text)
            
        except Exception as e:
            self.detection_complete.emit(None, f"检测错误: {str(e)}")

class CADDetectionGUI(QMainWindow):
    """CAD图纸检测系统主界面"""
    
    def __init__(self):
        super().__init__()
        self.model_path = None
        self.current_image = None
        self.detection_thread = None
        
        self.init_ui()
        self.load_default_model()
        
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("基于YOLO11的CAD图纸检测系统")
        self.setGeometry(100, 100, 1400, 900)
        
        # 创建主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # 创建主布局
        main_layout = QHBoxLayout(main_widget)
        
        # 左侧控制面板
        self.create_control_panel(main_layout)
        
        # 右侧显示区域
        self.create_display_area(main_layout)
        
        # 设置样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #1565C0;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #ccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
    def create_control_panel(self, parent_layout):
        """创建左侧控制面板"""
        control_panel = QVBoxLayout()
        
        # 模型选择组
        model_group = QGroupBox("模型配置")
        model_layout = QVBoxLayout()
        
        self.model_label = QLabel("当前模型: 未加载")
        self.model_label.setWordWrap(True)
        model_layout.addWidget(self.model_label)
        
        load_model_btn = QPushButton("选择模型文件")
        load_model_btn.clicked.connect(self.load_model)
        model_layout.addWidget(load_model_btn)
        
        model_group.setLayout(model_layout)
        control_panel.addWidget(model_group)
        
        # 检测参数组
        param_group = QGroupBox("检测参数")
        param_layout = QVBoxLayout()
        
        # 置信度阈值
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("置信度阈值:"))
        self.conf_spinbox = QDoubleSpinBox()
        self.conf_spinbox.setRange(0.1, 1.0)
        self.conf_spinbox.setValue(0.5)
        self.conf_spinbox.setSingleStep(0.05)
        conf_layout.addWidget(self.conf_spinbox)
        param_layout.addLayout(conf_layout)
        
        param_group.setLayout(param_layout)
        control_panel.addWidget(param_group)
        
        # 图像操作组
        image_group = QGroupBox("图像操作")
        image_layout = QVBoxLayout()
        
        load_image_btn = QPushButton("加载图像")
        load_image_btn.clicked.connect(self.load_image)
        image_layout.addWidget(load_image_btn)
        
        detect_btn = QPushButton("开始检测")
        detect_btn.clicked.connect(self.start_detection)
        image_layout.addWidget(detect_btn)
        
        save_btn = QPushButton("保存结果")
        save_btn.clicked.connect(self.save_result)
        image_layout.addWidget(save_btn)
        
        image_group.setLayout(image_layout)
        control_panel.addWidget(image_group)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        control_panel.addWidget(self.progress_bar)
        
        # 检测结果
        result_group = QGroupBox("检测结果")
        result_layout = QVBoxLayout()
        
        self.result_text = QTextEdit()
        self.result_text.setMaximumHeight(200)
        self.result_text.setReadOnly(True)
        result_layout.addWidget(self.result_text)
        
        result_group.setLayout(result_layout)
        control_panel.addWidget(result_group)
        
        # 添加弹簧
        control_panel.addStretch()
        
        # 创建左侧容器
        left_widget = QWidget()
        left_widget.setMaximumWidth(300)
        left_widget.setLayout(control_panel)
        parent_layout.addWidget(left_widget)
        
    def create_display_area(self, parent_layout):
        """创建右侧显示区域"""
        display_layout = QVBoxLayout()
        
        # 创建标签页
        self.tab_widget = QTabWidget()
        
        # 原始图像标签页
        self.original_label = QLabel("请加载CAD图纸图像")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setStyleSheet("background-color: white; border: 1px solid #ccc;")
        self.original_label.setMinimumSize(800, 600)
        self.tab_widget.addTab(self.original_label, "原始图像")
        
        # 检测结果标签页
        self.result_label = QLabel("检测结果将在此显示")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("background-color: white; border: 1px solid #ccc;")
        self.result_label.setMinimumSize(800, 600)
        self.tab_widget.addTab(self.result_label, "检测结果")
        
        display_layout.addWidget(self.tab_widget)
        
        # 创建右侧容器
        right_widget = QWidget()
        right_widget.setLayout(display_layout)
        parent_layout.addWidget(right_widget)
        
    def load_default_model(self):
        """加载默认模型"""
        # 检查是否存在训练好的模型
        model_path = "runs/train/cad_detection_exp/weights/best.pt"
        if os.path.exists(model_path):
            self.model_path = model_path
            self.model_label.setText(f"当前模型: {os.path.basename(model_path)}")
        else:
            self.model_label.setText("当前模型: 未找到训练好的模型，请手动选择")
            
    def load_model(self):
        """加载模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "选择YOLO模型文件", 
            "", 
            "PyTorch模型 (*.pt);;所有文件 (*.*)"
        )
        
        if file_path:
            self.model_path = file_path
            self.model_label.setText(f"当前模型: {os.path.basename(file_path)}")
            
    def load_image(self):
        """加载图像文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "选择CAD图纸图像", 
            "", 
            "图像文件 (*.png *.jpg *.jpeg *.bmp *.tiff);;所有文件 (*.*)"
        )
        
        if file_path:
            self.current_image = file_path
            
            # 显示原始图像
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                # 缩放图像以适应标签
                scaled_pixmap = pixmap.scaled(
                    self.original_label.size(), 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                self.original_label.setPixmap(scaled_pixmap)
                self.original_label.setText("")
                
                # 清空结果
                self.result_label.clear()
                self.result_label.setText("检测结果将在此显示")
                self.result_text.clear()
                
    def start_detection(self):
        """开始检测"""
        if not self.model_path:
            self.result_text.setText("请先加载模型文件")
            return
            
        if not self.current_image:
            self.result_text.setText("请先加载图像")
            return
            
        if not os.path.exists(self.model_path):
            self.result_text.setText("模型文件不存在")
            return
            
        # 禁用按钮
        self.set_detection_enabled(False)
        
        # 显示进度条
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # 创建检测线程
        self.detection_thread = DetectionThread(
            self.model_path,
            self.current_image,
            self.conf_spinbox.value()
        )
        
        self.detection_thread.progress_update.connect(self.update_progress)
        self.detection_thread.detection_complete.connect(self.detection_finished)
        self.detection_thread.start()
        
    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)
        
    def detection_finished(self, result_image, result_text):
        """检测完成回调"""
        self.progress_bar.setVisible(False)
        self.set_detection_enabled(True)
        
        if result_image is not None:
            # 显示检测结果，将numpy数组转换为QImage
            height, width, channel = result_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(
                result_image.data, 
                width, 
                height, 
                bytes_per_line, 
                QImage.Format_RGB888
            ).rgbSwapped()
            #转化为QPixmap并显示
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(
                self.result_label.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            self.result_label.setPixmap(scaled_pixmap)
            
            self.result_label.setText("")
            
            # 显示检测文本
            self.result_text.setText(result_text)
            
            # 切换到结果标签页
            self.tab_widget.setCurrentIndex(1)
        else:
            self.result_text.setText(result_text)
            
    def set_detection_enabled(self, enabled):
        """设置检测相关按钮状态"""
        for child in self.findChildren(QPushButton):
            if child.text() in ["开始检测", "加载图像", "选择模型文件", "保存结果"]:
                child.setEnabled(enabled)
                
    def save_result(self):
        """保存检测结果"""
        if self.result_label.pixmap() is None:
            self.result_text.setText("没有可保存的结果")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "保存检测结果", 
            "detection_result.png", 
            "PNG图像 (*.png);;JPEG图像 (*.jpg);;所有文件 (*.*)"
        )
        
        if file_path:
            self.result_label.pixmap().save(file_path)
            self.result_text.setText(f"结果已保存到: {file_path}")

def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用程序样式
    app.setStyle('Fusion')
    
    # 创建主窗口
    window = CADDetectionGUI()
    window.show()
    
    return app.exec()

if __name__ == '__main__':
    main()
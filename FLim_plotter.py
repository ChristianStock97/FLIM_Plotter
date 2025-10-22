from qtpy.QtWidgets import QLabel, QPushButton, QWidget, QGridLayout, QFileDialog, QSlider, QCheckBox, QMessageBox
from qtpy.QtCore import Qt
from time import sleep, time
import napari
import tiffile as tif
import numpy as np
from numba import cuda
import traceback
import logging
import gc
from global_fit import flim_fit_2D, mean_cuda_2D, dilate_cuda_2D, dilation_cuda_3D, mean_cuda_3D
from TO_HSV import flim_to_rgb
import configparser
import os

class Adaptive_GUI(QWidget):
    def __init__(self, viewer = None, parent=None):
        super().__init__(parent)
        self.viewer = viewer

        self.config = configparser.ConfigParser()
        self.config['Starting Parameters'] = {
        'Kernel': '0',
        'Threshold': '0',
        'MinTau': '0',
        'MaxTau': '1',
        'StartSample': '0'
        }

        self.config_path = "config.ini"
        if os.path.exists(self.config_path):
            print("Reading Config")
            self.config.read(self.config_path)
        else:
            print("Creating new Config")
            with open(self.config_path, 'w') as f:
                self.config.write(f)

        self.video_range = [0, 0]
        self.tau_range = [0.0, 3.0]
        self.threshold = 0.0
        self.kernel = 0
        self.time_bin = 1
        self.start_sample = 0
        self.stack_cpu = None
        self.ret_image_cpu = None
        self.dilated_stack = None
        self.tau_cpu = None
        self.RGB = None

        self.stack_gpu = None
        self.diluted_image_gpu = None
        self.tau_gpu = None
        # try to allocate common GPU buffers if CUDA is available
        try:
            self.cuda_available = cuda.is_available()
        except Exception:
            self.cuda_available = False

        self.pos_taus_gpu = None
        if self.cuda_available:
            try:
                self.pos_taus_gpu = cuda.to_device(np.linspace(0.5, 2.5, 1000, dtype=np.float32))
            except Exception:
                logging.exception("Failed to allocate pos_taus_gpu on device")
                self.pos_taus_gpu = None
                self.cuda_available = False
        else:
            # clear any device usage
            self.pos_taus_gpu = None
        self.x_data_gpu = None
        self.res_image_gpu = None

        self.block = (512)
        self.threads = (512)

    def smart_window(self):
        self.show_tau_check = QCheckBox("Tau")
        self.show_TimeStack_check = QCheckBox("Time-Stack")

        self.load_button = QPushButton("load Stack",self)
        self.load_button.setFixedSize(200, 30)
        self.load_button.setEnabled(True)
        self.load_button.clicked.connect(self.browse_timestack)

        self.Run_button = QPushButton("Run Fit")
        self.Run_button.setFixedSize(200, 30)
        self.Run_button.clicked.connect(self.run_Fit)
        if not getattr(self, 'cuda_available', False):
            # disable run button when CUDA not available and inform the user
            self.Run_button.setEnabled(False)
            self.cuda_label = QLabel("CUDA not available — GPU kernels disabled", self)
            self.cuda_label.setStyleSheet('color: red')
        else:
            self.cuda_label = QLabel("CUDA available", self)

        self.kernel_arv = QLabel(f"Kernel [{self.kernel}]", self)
        self.kernel_sl = QSlider()
        self.kernel_sl.setOrientation(Qt.Horizontal)
        self.kernel_sl.setTickInterval(1)
        self.kernel_sl.setRange(0,10)
        self.kernel_sl.valueChanged.connect(self.change_kernel_Value)
        self.kernel_sl.setValue(int(self.config['Starting Parameters']['Kernel']))

        self.timebin_arv = QLabel(f"Time_Bin [{self.time_bin}]", self)
        self.timebin_sl = QSlider()
        self.timebin_sl.setOrientation(Qt.Horizontal)
        self.timebin_sl.setTickInterval(1)
        self.timebin_sl.setRange(1,200)
        self.timebin_sl.valueChanged.connect(self.change_timebin_Value)
        self.timebin_sl.setValue(int(self.config['Starting Parameters']['Time_Bin']))


        self.threshold_arv = QLabel(f"Threshold [{self.threshold}]", self)
        self.thrshold_sl = QSlider()
        self.thrshold_sl.setOrientation(Qt.Horizontal)
        self.thrshold_sl.setTickInterval(1)
        self.thrshold_sl.setRange(0,2000)
        self.thrshold_sl.valueChanged.connect(self.change_threshold_Value)
        self.thrshold_sl.setValue(int(self.config['Starting Parameters']['Threshold']))

        self.minTau_label = QLabel("min Tau [0.0]", self)
        self.minTau_sl = QSlider()
        self.minTau_sl.setOrientation(Qt.Horizontal)
        self.minTau_sl.setTickInterval(1)
        self.minTau_sl.setRange(0,30)
        self.minTau_sl.valueChanged.connect(self.set_min_tau)
        self.minTau_sl.setValue(int(self.config['Starting Parameters']['MinTau']))

        self.maxTau_label = QLabel("max Tau [3.0]", self)
        self.maxTau_sl = QSlider()
        self.maxTau_sl.setOrientation(Qt.Horizontal)
        self.maxTau_sl.setTickInterval(1)
        self.maxTau_sl.setRange(0,30)
        self.maxTau_sl.valueChanged.connect(self.set_max_tau)
        self.maxTau_sl.setValue(int(self.config['Starting Parameters']['MaxTau']))

        self.start_pos_label = QLabel("Start [0]", self)
        self.start_pos__sl = QSlider()
        self.start_pos__sl.setOrientation(Qt.Horizontal)
        self.start_pos__sl.setTickInterval(1)
        self.start_pos__sl.setRange(0,0)
        self.start_pos__sl.valueChanged.connect(self.set_start)
        self.start_pos__sl.setValue(1)

        self.stop_pos_label = QLabel("Stop [0]", self)
        self.stop_pos__sl = QSlider()
        self.stop_pos__sl.setOrientation(Qt.Horizontal)
        self.stop_pos__sl.setTickInterval(1)
        self.stop_pos__sl.setRange(0,0)
        self.stop_pos__sl.valueChanged.connect(self.set_stop)
        self.stop_pos__sl.setValue(0)

        self.startSample_label = QLabel("Sample Offset [0]", self)
        self.startSample_sl = QSlider()
        self.startSample_sl.setOrientation(Qt.Horizontal)
        self.startSample_sl.setTickInterval(1)
        self.startSample_sl.setRange(0,32)
        self.startSample_sl.valueChanged.connect(self.set_startSample)
        self.startSample_sl.setValue(int(self.config['Starting Parameters']['StartSample']))

        layout = QGridLayout()
        layout.addWidget(self.show_tau_check, 0,0)
        layout.addWidget(self.show_TimeStack_check, 1, 0)
        layout.addWidget(self.load_button, 0, 1)
        layout.addWidget(self.Run_button, 1, 1)
        layout.addWidget(self.kernel_arv, 2, 0)
        layout.addWidget(self.kernel_sl, 2, 1)
        layout.addWidget(self.timebin_arv, 3, 0)
        layout.addWidget(self.timebin_sl, 3, 1)


        layout.addWidget(self.threshold_arv, 4, 0)
        layout.addWidget(self.thrshold_sl, 4, 1)
        layout.addWidget(self.minTau_label, 5, 0)
        layout.addWidget(self.minTau_sl, 5, 1)
        layout.addWidget(self.maxTau_label, 6, 0)
        layout.addWidget(self.maxTau_sl, 6, 1)
        layout.addWidget(self.startSample_label, 7, 0)
        layout.addWidget(self.startSample_sl, 7, 1)
        layout.addWidget(self.start_pos_label, 8, 0)
        layout.addWidget(self.start_pos__sl, 8, 1)
        layout.addWidget(self.stop_pos_label, 9, 0)
        layout.addWidget(self.stop_pos__sl, 9, 1)
        layout.addWidget(self.cuda_label, 10, 0, 1, 2)

        mda_tab = QWidget()
        mda_tab.setLayout(layout)
        mda_tab.setWindowTitle("Adaptive")
        return mda_tab

    def save_confog_values(self):
        self.config['Starting Parameters']['Kernel'] = str(self.kernel_sl.value())
        self.config['Starting Parameters']['Threshold'] = str(self.thrshold_sl.value())
        self.config['Starting Parameters']['MinTau'] = str(self.minTau_sl.value())
        self.config['Starting Parameters']['MaxTau'] = str(self.maxTau_sl.value())
        self.config['Starting Parameters']['StartSample'] = str(self.startSample_sl.value())
        with open(self.config_path, 'w') as f:
            self.config.write(f)

    def set_stop(self):
        self.video_range[1] = self.stop_pos__sl.value()
        self.stop_pos_label.setText(f"Stop [{self.video_range[1]}]")
        if self.video_range[1] <= self.video_range[0] and self.video_range[1] > 0:
            self.video_range[0] = self.video_range[1]-1
            self.start_pos_label.setText(f"Start [{self.video_range[0]}]")
            self.start_pos__sl.setValue(self.video_range[0])

    def set_start(self):
        self.video_range[0] = self.start_pos__sl.value()
        self.start_pos_label.setText(f"Start [{self.video_range[0]}]")
        if self.video_range[0] >= self.video_range[1] and self.video_range[0] < self.stack_cpu.shape[0]:
            self.video_range[1] = self.video_range[0] + 1
            self.stop_pos_label.setText(f"Stop [{self.video_range[1]}]")
            self.stop_pos__sl.setValue(int(self.video_range[1]))

    def run_Fit(self):
        if self.stack_cpu is not None:
            self.set_startSample()
            t = time()
            self.calculate_flim()
            print(f"Needed {time()-t} Seconds for fitting and arranging")
            self.save_confog_values()

    def set_startSample(self):
        self.start_sample = int(self.startSample_sl.value())
        self.startSample_label.setText(f"Sample Offset [{int(self.startSample_sl.value())}]")
        if self.stack_cpu is not None:
            start = 0
            if len(self.stack_cpu.shape) == 3:
                stop = 0.31 * (self.stack_cpu.shape[0]-1)
                steps = self.stack_cpu.shape[0]
            if len(self.stack_cpu.shape) == 4:
                stop = 0.31 * (self.stack_cpu.shape[1]-1)
                steps = self.stack_cpu.shape[1]
            # safely re-create the x_data_gpu device array
            try:
                if hasattr(self, 'x_data_gpu') and self.x_data_gpu is not None:
                    try:
                        del self.x_data_gpu
                    except Exception:
                        pass
                gc.collect()
                if getattr(self, 'cuda_available', False):
                    self.x_data_gpu = cuda.to_device(np.linspace(start, stop, steps, dtype=np.float32))
                else:
                    self.x_data_gpu = None
            except Exception:
                logging.exception("Failed to create x_data_gpu")
                self.x_data_gpu = None

    def set_max_tau(self):
        self.tau_range[1] = round(float(self.maxTau_sl.value()) / 10, 1)
        self.maxTau_label.setText(f"max Tau [{self.tau_range[1]}]")
        if self.tau_range[1] <= self.tau_range[0] and self.tau_range[1] > 0.0:
            self.tau_range[0] = round(self.tau_range[1]-0.1, 1)
            self.minTau_label.setText(f"min Tau [{self.tau_range[0]}]")
            self.minTau_sl.setValue(int(10*self.tau_range[0]))

    def set_min_tau(self):
        self.tau_range[0] = round(float(self.minTau_sl.value()) / 10, 1)
        self.minTau_label.setText(f"min Tau [{self.tau_range[0]}]")
        if self.tau_range[0] >= self.tau_range[1] and self.tau_range[0] < 3.0:
            self.tau_range[1] = round(self.tau_range[0] + 0.1, 1)
            self.maxTau_label.setText(f"max Tau [{self.tau_range[1]}]")
            self.maxTau_sl.setValue(int(self.tau_range[1]*10))

    def change_threshold_Value(self):
        self.threshold = int(self.thrshold_sl.value())
        self.threshold_arv.setText(f"Threshold [{self.threshold}]")

    def change_kernel_Value(self):
        self.kernel = int(self.kernel_sl.value())
        self.kernel_arv.setText(f"Kernel [{self.kernel}]")

    def change_timebin_Value(self):
        self.time_bin = int(self.timebin_sl.value())
        self.timebin_arv.setText(f"Time_Bin [{self.time_bin}]")

    def browse_timestack(self):
        waveform_name, ok = QFileDialog.getOpenFileName(
            self,
            "Select a File",
            "E:",
            "Flim Stack (*.tif)")
        
        if ok:
            # read file with error handling
            try:
                tmp: np.array = tif.imread(waveform_name)
            except Exception as e:
                tb = traceback.format_exc()
                QMessageBox.critical(self, "File Read Error", f"Unable to read TIFF file:\n{e}\n\nSee console for details.")
                logging.error(tb)
                self.stack_cpu = None
                return

            if not isinstance(tmp, np.ndarray):
                QMessageBox.warning(self, "Invalid Data", "Loaded file does not contain a numeric array.")
                self.stack_cpu = None
                return

            if tmp.size == 0:
                QMessageBox.warning(self, "Empty File", "Selected TIFF contains no data.")
                self.stack_cpu = None
                return

            # update threads based on image width if available
            try:
                self.threads = (tmp.shape[-1])
            except Exception:
                self.threads = (512)

            if len(tmp.shape) == 3:
                try:
                    t = time()
                    tmp = tmp[self.start_sample:,:,:]
                    # normalize safely
                    maxv = tmp.max() if tmp.size and np.isfinite(tmp.max()) else 1
                    tmp =  (256 * tmp / maxv).astype(np.uint8)
                    self.stack_cpu = np.reshape(tmp, (1,tmp.shape[0], tmp.shape[1], tmp.shape[2]))
                    dt = time()
                    print(f"Neede {dt-t} Seconds for Data read")
                    self.stop_pos__sl.setRange(1,self.stack_cpu.shape[0])
                    self.stop_pos__sl.setValue(self.stack_cpu.shape[0])
                    self.start_pos__sl.setRange(0,self.stack_cpu.shape[0]-1)
                    self.start_pos__sl.setValue(0)
                except Exception as e:
                    QMessageBox.critical(self, "Data Error", f"Failed to prepare 3D stack:\n{e}")
                    logging.exception("Failed to prepare 3D stack")
                    self.stack_cpu = None
                    return

            elif len(tmp.shape) == 4:
                try:
                    t = time()
                    # assume (Frames, Samples, H, W)
                    self.stack_cpu = tmp[:,self.start_sample:,:,:].astype(np.uint8)
                    dt = time()
                    print(f"Neede {dt-t} Seconds for Data read")
                    self.stop_pos__sl.setRange(1,self.stack_cpu.shape[0])
                    self.stop_pos__sl.setValue(self.stack_cpu.shape[0])
                    self.start_pos__sl.setRange(0,self.stack_cpu.shape[0]-1)
                    self.start_pos__sl.setValue(0)
                except Exception as e:
                    QMessageBox.critical(self, "Data Error", f"Failed to prepare 4D stack:\n{e}")
                    logging.exception("Failed to prepare 4D stack")
                    self.stack_cpu = None
                    return

            else:
                QMessageBox.warning(self, "Unsupported Shape", f"Unsupported TIFF shape: {tmp.shape}")
                self.stack_cpu = None
                return

            self.set_startSample()
        
        else:
            self.stack_cpu = None

    def calculate_flim(self):

        show_tau = self.show_tau_check.isChecked()
        show_TimeStack = self.show_TimeStack_check.isChecked()
        print(f"Checkboxes set: {(show_tau, show_TimeStack)}")
        self.viewer.layers.clear()

        if self.stack_cpu is None:
            QMessageBox.warning(self, "No data", "No stack loaded. Please load a TIFF stack first.")
            return

        if not getattr(self, 'cuda_available', False):
            QMessageBox.critical(self, "CUDA required", "CUDA is required to run the GPU kernels. Aborting.")
            return

        try:
            pos_taus_gpu = cuda.to_device(np.linspace(self.tau_range[0], self.tau_range[1], 1000, dtype=np.float32))
        except Exception as e:
            logging.exception("Failed to allocate pos_taus_gpu in calculate_flim")
            QMessageBox.critical(self, "CUDA error", f"Unable to allocate device memory:\n{e}")
            return

        if self.stack_cpu.shape[0] > 1:
            image_per_calculation = self.time_bin
            s = [image_per_calculation, self.stack_cpu.shape[1], self.stack_cpu.shape[2], self.stack_cpu.shape[3]]
            stack_gpu = cuda.to_device(np.zeros(s, np.float32))
            n_images = (self.video_range[1]- self.video_range[0]) // image_per_calculation + 1
        else:
            stack_gpu = cuda.to_device(np.zeros(self.stack_cpu.shape[1:], np.float32))
            n_images = 1

        diluted_image_gpu = cuda.to_device(np.zeros(self.stack_cpu.shape[1:], np.float32))
        res_image_gpu = cuda.to_device(np.zeros(self.stack_cpu.shape[2:], np.float32))
        tau_gpu = cuda.to_device(np.zeros(self.stack_cpu.shape[2:], np.float32))
        rbg_gpu = cuda.to_device(np.zeros((self.stack_cpu.shape[2], self.stack_cpu.shape[3], 3), dtype=np.uint8))
        
        try:
            self.viewer.layers["Intensity"].data = np.zeros((n_images, self.stack_cpu.shape[2], self.stack_cpu.shape[3]), dtype=np.float32)
        except:
            self.viewer.add_image(np.zeros((n_images, self.stack_cpu.shape[2], self.stack_cpu.shape[3]), dtype=np.float32), name = "Intensity")
        
        if show_TimeStack:
            try:
                self.viewer.layers["Dilated"].data = np.zeros(self.stack_cpu.shape, np.float32)
            except:
                self.viewer.add_image(np.zeros(self.stack_cpu.shape, np.float32), name = "Dilated")

        if show_tau:
            try:
                self.viewer.layers["Tau"].data = np.zeros((n_images, self.stack_cpu.shape[2], self.stack_cpu.shape[3]), dtype=np.float32)
            except:
                self.viewer.add_image(np.zeros((n_images, self.stack_cpu.shape[2], self.stack_cpu.shape[3]), dtype=np.float32), name = "Tau")
        
        try:
            self.viewer.layers["RGB"].data = np.zeros((n_images, self.stack_cpu.shape[2], self.stack_cpu.shape[3], 3), dtype=np.uint8)
        except:
            self.viewer.add_image(np.zeros((n_images, self.stack_cpu.shape[2], self.stack_cpu.shape[3], 3), dtype=np.uint8), name = "RGB")

        if self.stack_cpu.shape[0] > 1:
            stream = cuda.stream()
            t = time()  
            image_idx = 0

            for n in range(self.video_range[0], self.video_range[1], image_per_calculation):
                stack_gpu = cuda.to_device(self.stack_cpu[n:n+image_per_calculation], stream=stream)
                dilation_cuda_3D[self.block, self.threads](diluted_image_gpu, stack_gpu, self.kernel, 0)
                flim_fit_2D[self.block, self.threads](tau_gpu, diluted_image_gpu, pos_taus_gpu, self.x_data_gpu)
                mean_cuda_3D[self.block, self.threads](res_image_gpu, stack_gpu)
                flim_to_rgb[self.block, self.threads](rbg_gpu, tau_gpu, res_image_gpu, self.tau_range[0], self.tau_range[1], self.threshold)

                res_image_gpu.copy_to_host(self.viewer.layers["Intensity"].data[image_idx], stream=stream)
                if show_TimeStack:
                    diluted_image_gpu.copy_to_host(self.viewer.layers["Dilated"].data[image_idx], stream=stream)
                if show_tau:
                    tau_gpu.copy_to_host(self.viewer.layers["Tau"].data[image_idx], stream=stream)
                rbg_gpu.copy_to_host(self.viewer.layers["RGB"].data[image_idx], stream=stream)
                image_idx += 1
            self.viewer.layers["Intensity"].refresh()
            if show_TimeStack:
                self.viewer.layers["Dilated"].refresh()
            if show_tau:
                self.viewer.layers["Tau"].refresh()
            self.viewer.layers["RGB"].refresh()
            print(f"calculated {(time()-t)/(self.video_range[1] - self.video_range[0])} seconds per Image")

        elif self.stack_cpu.shape[0] == 1:
            stream = cuda.stream()
            t = time()

            for n in range(self.video_range[0], self.video_range[1], 3):
                stack_gpu = cuda.to_device(self.stack_cpu, stream=stream)
                dilate_cuda_2D[self.block, self.threads](diluted_image_gpu, stack_gpu, self.kernel)
                flim_fit_2D[self.block, self.threads](tau_gpu, diluted_image_gpu, pos_taus_gpu, self.x_data_gpu)
                mean_cuda_2D[self.block, self.threads](res_image_gpu, stack_gpu)
                flim_to_rgb[self.block, self.threads](rbg_gpu, tau_gpu, res_image_gpu, self.tau_range[0], self.tau_range[1], self.threshold)

                res_image_gpu.copy_to_host(self.viewer.layers["Intensity"].data[n], stream=stream)
                if show_TimeStack:
                    diluted_image_gpu.copy_to_host(self.viewer.layers["Dilated"].data[n], stream=stream)
                if show_tau:
                    tau_gpu.copy_to_host(self.viewer.layers["Tau"].data[n], stream=stream)
                rbg_gpu.copy_to_host(self.viewer.layers["RGB"].data[n], stream=stream)

            self.viewer.layers["Intensity"].refresh()
            if show_TimeStack:
                self.viewer.layers["Dilated"].refresh()
            if show_tau:
                self.viewer.layers["Tau"].refresh()
            self.viewer.layers["RGB"].refresh()
            print(f"calculated {(time()-t)/(self.video_range[1] - self.video_range[0])} seconds per Image")

        else:
            print(f"CPU stack has wrong dimensions: {self.stack_cpu.shape}")


def test_gui():
    viewer = napari.Viewer()
    smart = Adaptive_GUI(viewer=viewer)
    viewer.window.add_dock_widget(smart.smart_window(),area='left')
    napari.run()

if __name__ == "__main__":
    test_gui()
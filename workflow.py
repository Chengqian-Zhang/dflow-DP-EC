from dflow.python import upload_packages
import tempfile
tempfile.tempdir = "/data/tmp"
import json,pathlib
from typing import List
from dflow import (
    Workflow,
    Step,
    argo_range,
    SlurmRemoteExecutor,
    upload_artifact,
    download_artifact,
    InputArtifact,
    OutputArtifact,
    ShellOPTemplate
)
from dflow.python import (
    PythonOPTemplate,
    OP,
    OPIO,
    OPIOSign,
    Artifact,
    Slices,
    upload_packages
)
import time

import subprocess, os, shutil, glob, ase.io
from pathlib import Path
from typing import List
from monty.serialization import loadfn
from dflow.plugins.bohrium import BohriumContext, BohriumExecutor
from dflow.plugins.dispatcher import DispatcherExecutor
from monty.serialization import loadfn

from toolkit.analysis.band_align import BandAlign
from toolkit.analysis.band_align import get_alignment
from toolkit.utils.utils import au2eV
from toolkit.analysis.atom_density import AtomDensity
import numpy as np
import matplotlib.pyplot as plt

def get_combined_file(hartree_list, f_name):
    _width_list = hartree_list.columns.to_numpy(dtype=float)
    _hartree_list = hartree_list.mean().to_numpy()
    combined_wdith_hartree = np.stack([_width_list, _hartree_list])
    np.savetxt(f_name, combined_wdith_hartree.T)

def parse_mdev_file(mdev_file):
    steps =  np.loadtxt(mdev_file, usecols=0)
    max_mdevs = np.loadtxt(mdev_file, usecols=4)
    return steps, max_mdevs

def parse_density_file(density_file):
    z = np.loadtxt(density_file, usecols=0)
    den = np.loadtxt(density_file, usecols=1)
    return z, den

def get_density_dev(density_file_list):
    den_list = []
    for density_file in density_file_list:
        z, den = parse_density_file(density_file)
        den_list.append(den)
    den_list = np.array(den_list)
    den_dev = den_list.std(axis=0)
    den_mean = den_list.mean(axis=0)
    return den_mean, den_dev, z

class lammps(OP):
    """
    class for run MD in lammps
    """
    def __init__(self,infomode=1):
        self.infomode = infomode

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            'input_lammps': Artifact(Path),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            'output_lammps': Artifact(Path)                                                                                                             
        })

    @OP.exec_sign_check
    def execute(self, op_in: OPIO) -> OPIO:
        cwd = os.getcwd()
        os.chdir(op_in["input_lammps"])
        cmd = "lmp -i in.lammps"
        subprocess.call(cmd, shell=True)
        os.chdir(cwd)
        op_out = OPIO({
            "output_lammps": op_in["input_lammps"]
        })
        return op_out

class modeldevi(OP):
    '''
    class for model_devi
    '''
    def __init__(self,infomode=1):
        self.infomode = infomode

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            'input_devi': Artifact(Path),
            'nmodel': int
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            'output_devi': Artifact(Path)                                                                                                               
        })

    @OP.exec_sign_check
    def execute(self, op_in: OPIO) -> OPIO:
        cwd = os.getcwd()
        os.chdir(op_in["input_devi"])
        input_path = os.getcwd()
        devi_path = os.path.join(input_path,"02.model_devi_plots")
        os.makedirs(devi_path)
        devi_input_path = os.path.join(devi_path,"00.inputs")
        os.makedirs(devi_input_path)
        devi_output_path = os.path.join(devi_path,"01.outputs")
        os.makedirs(devi_output_path)
        os.chdir(devi_input_path)
        for ii in range(op_in["nmodel"]):
            os.system("ln -s ../../01.dpmd/02.outputs/%02d.run/ %02d.run"%(ii,ii))
        # here is inputs
        mdev_file_list = ["00.run/model_devi.out", "01.run/model_devi.out", "02.run/model_devi.out", "03.run/model_devi.out"]
        row = 1
        col = 1
        fig = plt.figure(figsize=(8*col,4.5*row), dpi=150, facecolor='white')
        gs = fig.add_gridspec(row,col)
        ax  = fig.add_subplot(gs[0])
        for mdev_file in mdev_file_list:
            steps, max_mdevs = parse_mdev_file(mdev_file=mdev_file)
            ax.scatter(steps, max_mdevs, s=5)

            ax.set_ylabel("Max Force Model Deviation [eV/A]")
            ax.set_xlabel("Steps")
            ax.tick_params(direction='in')
        fig.savefig("../01.outputs/model_devi_plot.png", dpi=400)

        os.chdir(cwd)
        op_out = OPIO({
            "output_devi": op_in["input_devi"]
        })
        return op_out

class densityprofile(OP):
    '''
    class for O_density_profile
    '''
    def __init__(self,infomode=1):
        self.infomode = infomode

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            'input_density': Artifact(Path),
            'nmodel': int
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            'output_density': Artifact(Path)                                                                                                            
        })

    @OP.exec_sign_check
    def execute(self, op_in: OPIO) -> OPIO:
        cwd = os.getcwd()
        os.chdir(op_in["input_density"])
        input_path = os.getcwd()
        density_path = os.path.join(input_path,"03.o_density_profile")
        os.makedirs(density_path)
        density_input_path = os.path.join(density_path,"00.inputs")
        os.makedirs(density_input_path)
        density_output_path = os.path.join(density_path,"01.outputs")
        os.makedirs(density_output_path)
        os.chdir(density_input_path)
        for ii in range(op_in["nmodel"]):
            os.system("ln -s ../../01.dpmd/02.outputs/%02d.run/ %02d.run"%(ii,ii))
        traj_file_list = ["00.run/sno2-water.xyz", "01.run/sno2-water.xyz", "02.run/sno2-water.xyz", "03.run/sno2-water.xyz"]
        surf2_Sn_idx = [592, 600, 610, 616, 617, 624, 625, 628, 629, 638, 639, 645, 652, 653, 660, 664]
        surf1_Sn_idx = [593, 595, 596, 599, 604, 606, 609, 613, 619, 633, 636, 642, 648, 656, 661, 663]
        cell = [12.745, 13.399, 40.985]
        O_idx = [287, 288, 289, 291, 292, 293, 294, 298, 301, 303, 304, 305, 307, 310, 311, 315, 320, 322, 325, 326, 330, 333, 336, 339, 340, 342, 345, 346, 347, 348, 349, 352, 358, 361, 363, 364, 365, 366, 367, 368, 370, 373, 376, 379, 382, 384, 385, 386, 388, 389, 391, 392, 393, 394, 395, 396, 397, 400, 403, 406, 408, 410, 411, 414, 417, 419, 420, 421, 422, 423, 424, 425, 428, 429, 430, 432, 435, 438, 441, 444, 445, 447, 452, 453, 455, 458, 460, 461, 463, 464, 465, 466, 469, 472, 473, 474, 478, 481, 485, 487, 488, 490, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 522, 523, 525, 528, 531, 534, 537, 539, 540, 541, 542, 547, 548, 550, 553, 556, 558, 561, 564, 567, 570, 571, 572, 573, 574, 575, 578, 579, 580, 583, 584, 585, 586, 587]

        # process trajectory
        for idx, traj_file in enumerate(traj_file_list):
            inp_dict={
                "xyz_file": traj_file,
                "cell": cell,
                "surf2": surf2_Sn_idx,
                "surf1": surf1_Sn_idx,
                "density_type":[
                    {
                        "element": "O",
                        "idx_method": "manual",
                        "idx_list": O_idx,
                        "density_unit": "water",
                        "dz": 0.05,
                        "name": f"O_density_{idx}"
                        }
                   ]
            }
            ad = AtomDensity(inp_dict)
            ad.run()
        # plot density
        density_file_list = ["O_density_0.dat", "O_density_1.dat", "O_density_2.dat", "O_density_3.dat"]

        row = 1
        col = 1
        fig = plt.figure(figsize=(12*col,6*row), dpi=150, facecolor='white')
        gs = fig.add_gridspec(row,col)
        ax  = fig.add_subplot(gs[0])
        den_mean, den_dev, z = get_density_dev(density_file_list)
        ax.plot(z, den_mean)
        ax.fill_between(z, den_mean+den_dev, den_mean-den_dev, color='red', alpha=0.5)
        ax.set_ylabel("O Density [g/cm3]")
        ax.set_xlabel("z ['Angstrom']")
        ax.tick_params(direction='in')
        fig.savefig("../01.outputs/density_profile.png", dpi=400)

        os.chdir(cwd)
        op_out = OPIO({
            "output_density": op_in["input_density"]
        })
        return op_out

class DFTtasks(OP):
    '''
    class for prepare DFT calculation tasks
    '''
    def __init__(self,infomode=1):
        self.infomode = infomode

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            'input_dfttasks': Artifact(Path),
            'nsample': int,
            'cp2k_input': Artifact(Path),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            'output_dfttasks': Artifact(List[Path])                                                                                                     
        })

    @OP.exec_sign_check
    def execute(self, op_in: OPIO) -> OPIO:
        cwd = os.getcwd()
        os.chdir(op_in["input_dfttasks"])
        input_path = os.getcwd()
        dft_path = os.path.join(input_path,"04.dft")
        os.makedirs(dft_path)
        dft_input_path = os.path.join(dft_path,"00.inputs")
        os.makedirs(dft_input_path)
        print(os.listdir())
        print(dft_input_path)
        os.chdir(dft_input_path)
        os.symlink("../../01.dpmd/02.outputs/00.run/sno2-water.xyz","sno2-water.xyz")

        shutil.copyfile(op_in["cp2k_input"],"template.inp")
        ## select conf from traj
        ls = ase.io.read("sno2-water.xyz",index=':')
        total_conf = len(ls)
        interval = int(total_conf/op_in["nsample"])
        tasks = []
        for i in range(op_in["nsample"]):
            task_path = os.path.join(dft_input_path,"task.%06d"%i)
            os.makedirs(task_path)
            tag = i*interval
            ase.io.write("task.%06d/single.xyz"%i,ls[tag])
            os.system("sed -i '1,2d' 'task.%06d/single.xyz'"%i)
            shutil.copyfile("template.inp","task.%06d/template.inp"%i)
            tasks.append(pathlib.Path(task_path))
        os.chdir(cwd)
        op_out = OPIO({
            "output_dfttasks": tasks
        })
        return op_out

class CP2K(OP):
    """
    class for CP2K calculation
    """
    def __init__(self,infomode=1):
        self.infomode = infomode

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            'input_cp2k': Artifact(Path),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            'output_cp2k': Artifact(Path)                                                                                                               
        })

    @OP.exec_sign_check
    def execute(self, op_in: OPIO) -> OPIO:
        cwd = os.getcwd()
        os.chdir(op_in["input_cp2k"])
        cmd = "bash -c \"ulimit -s unlimited && source /opt/intel/oneapi/setvars.sh && source /root/cp2k-7.1/tools/toolchain/install/setup && mpirun -n 32 --allow-run-as-root --oversubscribe /root/cp2k-7.1/exe/local/cp2k.popt -i template.inp -o output.out\""
        subprocess.call(cmd, shell=True)
        os.chdir(cwd)
        op_out = OPIO({
            "output_cp2k": op_in["input_cp2k"]
        })
        return op_out

class DFTpost(OP):
    '''
    class for collecting *.cube files
    '''
    def __init__(self,infomode=1):
        self.infomode = infomode

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            'input_dftpost': Artifact(Path),
            'nsample': int
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            'output_dftpost': Artifact(Path)                                                                                                            
        })

    @OP.exec_sign_check
    def execute(self, op_in: OPIO) -> OPIO:
        cwd = os.getcwd()
        os.chdir(op_in["input_dftpost"])
        input_path = os.getcwd()
        dft_path = os.path.join(input_path,"04.dft")
        dft_input_path = os.path.join(dft_path,"00.inputs")
        dft_output_path = os.path.join(dft_path,"01.outputs")
        os.makedirs(dft_output_path)
        for i in range(op_in["nsample"]):
            os.symlink(os.path.join(dft_input_path,"task.%06d"%i,"SnO2-v_hartree-1_0.cube"),os.path.join(dft_output_path,"SnO2-v_hartree-1_%d.cube"%(i+1)))
        os.chdir(cwd)
        op_out = OPIO({
            "output_dftpost": op_in["input_dftpost"]
        })
        return op_out

class hartree(OP):
    '''
    class for calculating hartree
    '''
    def __init__(self,infomode=1):
        self.infomode = infomode

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            'input_hartree': Artifact(Path),
            'nsample': int
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            'output_hartree': Artifact(Path)                                                                                                            
        })

    @OP.exec_sign_check
    def execute(self, op_in: OPIO) -> OPIO:
        cwd = os.getcwd()
        os.chdir(op_in["input_hartree"])
        input_path = os.getcwd()
        hartree_path = os.path.join(input_path,"05.solid_water_hartree")
        os.makedirs(hartree_path)
        hartree_input_path = os.path.join(hartree_path,"00.inputs")
        os.makedirs(hartree_input_path)
        hartree_output_path = os.path.join(hartree_path,"01.outputs")
        os.makedirs(hartree_output_path)
        os.chdir(hartree_input_path)
        os.system("ln -s ../../04.dft/01.outputs/ 00.hartrees")
        inp = {
            "input_type": "cube",
            "ave_param":{
                    "prefix": "00.hartrees/SnO2-v_hartree-1_",
                    "index": (1, op_in["nsample"]+1),
                    "l1": 3.45,
                    "l2": 3.45,
                    "ncov": 2,
                    "save": True,
                    "save_path":"../01.outputs"
            },
            "shift_param":{
                    "surf1_idx": [593, 595, 596, 599, 604, 606, 609, 613, 619, 633, 636, 642, 648, 656, 661, 663],
                    "surf2_idx": [592, 600, 610, 616, 617, 624, 625, 628, 629, 638, 639, 645, 652, 653, 660, 664]
            },
            "water_width_list": [5, 6, 7,  8, 9, 10, 11, 12, 13],
            "solid_width_list": [2, 3, 4, 5, 6, 7, 8, 9]
        }
        x = BandAlign(inp)
        get_combined_file(x.water_hartree_list, f_name="../01.outputs/water_hartree.dat")
        get_combined_file(x.solid_hartree_list, f_name="../01.outputs/solid_hartree.dat")
        os.chdir(cwd)
        op_out = OPIO({
            "output_hartree": op_in["input_hartree"]
        })
        return op_out

class bandalignment(OP):
    '''
    class for band alignment
    '''
    def __init__(self,infomode=1):
        self.infomode = infomode

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            'input_band': Artifact(Path),
            'nsample': int
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            'output_band': Artifact(Path)                                                                                                               
        })

    @OP.exec_sign_check
    def execute(self, op_in: OPIO) -> OPIO:
        cwd = os.getcwd()
        os.chdir(op_in["input_band"])
        input_path = os.getcwd()
        band_path = os.path.join(input_path,"06.plot_band_alignment")
        os.makedirs(band_path)
        band_input_path = os.path.join(band_path,"00.inputs")
        os.makedirs(band_input_path)
        band_output_path = os.path.join(band_path,"01.outputs")
        os.makedirs(band_output_path)
        os.chdir(band_input_path)
        os.system("ln -s ../../05.solid_water_hartree/01.outputs/solid_hartree.dat solid_hartree.dat")
        os.system("ln -s ../../05.solid_water_hartree/01.outputs/water_hartree.dat water_hartree.dat")
        bulk_vbm = 0.182771
        bulk_cbm = 0.222584

        solid_width_list = np.loadtxt("./solid_hartree.dat", usecols=0)
        solid_hartree_list = np.loadtxt("./solid_hartree.dat", usecols=1)
        water_width_list = np.loadtxt("./water_hartree.dat", usecols=0)
        water_hartree_list = np.loadtxt("./water_hartree.dat", usecols=1)
        water_hartree = water_hartree_list.max()
        solid_hartree = solid_hartree_list.max()
        vbm = get_alignment(level = bulk_vbm*au2eV, ref_hartree=water_hartree, ref_solid_hartree=solid_hartree, ref_bulk=True)
        cbm = get_alignment(level = bulk_cbm*au2eV, ref_hartree=water_hartree, ref_solid_hartree=solid_hartree, ref_bulk=True)

        with open("../01.outputs/band_alignment.dat", "w") as f:
            f.write(f"VBM (SHE): {vbm:2.3f} [V]\n")
            f.write(f"CBM (SHE): {cbm:2.3f} [V]\n")

        os.chdir(cwd)
        op_out = OPIO({
            "output_band": op_in["input_band"]
        })
        return op_out

def main():
    ## Organizing lammps input files
    n_model = 4
    nsample = 3
    cwd = os.getcwd()
    input_path = os.path.join(cwd,"01.dpmd","00.inputs")
    pot_path = os.path.join(cwd,"01.dpmd","01.pot")
    output_path = os.path.join(cwd,"01.dpmd","02.outputs")
    if(os.path.exists(output_path)):
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    for ii in range(n_model):
        task_path = os.path.join(output_path,"%02d.run"%ii)
        os.makedirs(task_path)
        os.symlink(os.path.join(input_path,"in.lammps-%d"%ii),os.path.join(task_path,"in.lammps"))
        os.symlink(os.path.join(input_path,"conf.lmp"),os.path.join(task_path,"conf.lmp"))
        for jj in range(n_model):
            os.symlink(os.path.join(pot_path,"c%03d.pb"%jj),os.path.join(task_path,"c%03d.pb"%jj))
    tasks = glob.glob(os.path.join(output_path,"*.run"))

    ## define a workflow
    wf = Workflow(name = "dpectutorial",context=brm_context)
    ## Step1 run MD by lammps
    '''
    如果在本地运行lammps,用这段代码
    lammps_tem = PythonOPTemplate(lammps,slices=Slices("{{item}}", input_artifact=["input_lammps"],output_artifact=["output_lammps"]),image="registry.dp.tech/dptech/deepmd-kit:2.1.5-cuda11.6",command=["python3"])
    md_lammps = Step("MD-LAMMPS",template=lammps_tem,artifacts={"input_lammps":upload_artifact(tasks)},with_param=argo_range(n_model),key="MD-LAMMPS-{{item}}",util_image="registry.dp.tech/dptech/deepmd-kit:2.1.5-cuda11.6",util_command=['python3'])
    wf.add(md_lammps)
    '''
    lammps_tem = PythonOPTemplate(lammps,slices=Slices("{{item}}", input_artifact=["input_lammps"],output_artifact=["output_lammps"]),image="registry.dp.tech/dptech/prod-11461/chengqian:v2",command=["python3"])
    md_lammps = Step("MD-LAMMPS",template=lammps_tem,artifacts={"input_lammps":upload_artifact(tasks)},with_param=argo_range(n_model),key="MD-LAMMPS-{{item}}",util_image="registry.dp.tech/dptech/prod-11461/chengqian:v2",util_command=['python3'],executor=BohriumExecutor(executor="bohrium_v2", extra={"scassType":"c32_m128_4 * NVIDIA V100","projectId": 1065,"jobType":"container", "logFiles": []}))
    wf.add(md_lammps)

    ## Step2 model_devi
    Modeldevi = Step(
        name="MODELDEVI",
        template=PythonOPTemplate(modeldevi,image="registry.dp.tech/dptech/prod-11461/chengqian:v2",command=["python3"]),
        artifacts={"input_devi":md_lammps.outputs.artifacts["output_lammps"]},
        parameters = {"nmodel":n_model}
        )
    wf.add(Modeldevi)

    ## Step3 density
    Density = Step(
        name="DENSITY",
        template=PythonOPTemplate(densityprofile,image="registry.dp.tech/dptech/prod-11461/chengqian:v2",command=["python3"]),
        artifacts={"input_density":Modeldevi.outputs.artifacts["output_devi"]},
        parameters = {"nmodel":n_model}
        )
    wf.add(Density)

    ## Step4 prepare cp2k tasks
    dfttasks = Step(
        name="DFTTASKS",
        template=PythonOPTemplate(DFTtasks,image="registry.dp.tech/dptech/prod-11461/chengqian:v2",command=["python3"]),
        artifacts={"input_dfttasks":md_lammps.outputs.artifacts["output_lammps"],
                   "cp2k_input":upload_artifact("template.inp")
                },
        parameters = {"nsample":nsample}
        )
    wf.add(dfttasks)

    ## Step5 run cp2k tasks
    cp2k = PythonOPTemplate(CP2K,slices=Slices("{{item}}", input_artifact=["input_cp2k"],output_artifact=["output_cp2k"]),image="toolkit_ase_cp2k",command=["python3"])
    cp2k_cal = Step("CP2K-Cal",template=cp2k,artifacts={"input_cp2k":dfttasks.outputs.artifacts["output_dfttasks"]},with_param=argo_range(nsample),key="CP2K-Cal-{{item}}",util_command=['python3'],executor=BohriumExecutor(executor="bohrium_v2", extra={"scassType":"c32_m128_cpu","projectId": 1065, "logFiles": []}))
    wf.add(cp2k_cal)

    ## Step6 collect *.cube files
    dftpost = Step(
        name="DFTPOST",
        template=PythonOPTemplate(DFTpost,image="registry.dp.tech/dptech/prod-11461/chengqian:v2",command=["python3"]),
        artifacts={"input_dftpost":cp2k_cal.outputs.artifacts["output_cp2k"]},
        parameters = {"nsample":nsample}
        )
    wf.add(dftpost)

    ## Step7 hartree
    Hartree = Step(
        name="HARTREE",
        template=PythonOPTemplate(hartree,image="registry.dp.tech/dptech/prod-11461/chengqian:v2",command=["python3"]),
        artifacts={"input_hartree":dftpost.outputs.artifacts["output_dftpost"]},
        parameters = {"nsample":nsample}
        )
    wf.add(Hartree)

    ## Step8 bandalignment
    Bandalignment = Step(
        name="BANDALIGNMENT",
        template=PythonOPTemplate(bandalignment,image="registry.dp.tech/dptech/prod-11461/chengqian:v2",command=["python3"]),
        artifacts={"input_band":Hartree.outputs.artifacts["output_hartree"]},
        parameters = {"nsample":nsample}
        )
    wf.add(Bandalignment)

    wf.submit()

    while wf.query_status() in ["Pending","Running"]:
        time.sleep(4)
    assert(wf.query_status() == 'Succeeded')
    step1 = wf.query_step(name="MD-LAMMPS")[0]
    download_artifact(step1.outputs.artifacts["output_lammps"])

    assert(wf.query_status() == 'Succeeded')
    step3 = wf.query_step(name="DENSITY")[0]
    download_artifact(step3.outputs.artifacts["output_density"])

    assert(wf.query_status() == 'Succeeded')
    step8 = wf.query_step(name="BANDALIGNMENT")[0]
    download_artifact(step8.outputs.artifacts["output_band"])

if __name__ == "__main__":
    brm_context = BohriumContext(
        executor="mixed",
        #executor="bohrium_v2",
        extra={"scass_type":"c32_m128_4 * NVIDIA V100","program_id":1065,"job_type":"container"}, # 全局bohrium配置
        username="",
        password=""
    )
    main()
    ## 后处理
    os.system("mv tmp/inputs/artifacts/input_devi/02.model_devi_plots ./")
    os.system("mv tmp/inputs/artifacts/input_devi/03.o_density_profile ./")
    os.system("mv tmp/inputs/artifacts/input_dftpost/04.dft ./")
    os.system("mv tmp/inputs/artifacts/input_dftpost/05.solid_water_hartree ./")
    os.system("mv tmp/inputs/artifacts/input_dftpost/06.plot_band_alignment ./")
    os.system("rm -r tmp")

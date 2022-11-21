from dflow import config, s3_config
config["host"] = ""
s3_config["endpoint"] = ""
config["k8s_api_server"] = ""
config["token"] = ""

from dflow.python import upload_packages
upload_packages.append("/usr/local/lib/python3.8/dist-packages/ase")

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
from monty.serialization import loadfn

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
        cmd = "ulimit -s unlimited && mpirun --allow-run-as-root --oversubscribe -n 32 cp2k.popt -i template.inp >> output.log"
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
            os.symlink(os.path.join(dft_input_path,"task.%06d"%i,"SnO2-v_hartree-1_0.cube"),os.path.join(dft_output_path,"SnO2-v_hartree-1_%d.cube"%i))
        os.chdir(cwd)
        op_out = OPIO({
            "output_dftpost": op_in["input_dftpost"]
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
    wf = Workflow(name = "dpectutorial",context=brm_context, host="https://workflow.dp.tech/")
    ## Step1 run MD by lammps
    '''
    如果在本地运行lammps,用这段代码
    lammps_tem = PythonOPTemplate(lammps,slices=Slices("{{item}}", input_artifact=["input_lammps"],output_artifact=["output_lammps"]),image="registry.dp.tech/dptech/deepmd-kit:2.1.5-cuda11.6",command=["python3"])
    md_lammps = Step("MD-LAMMPS",template=lammps_tem,artifacts={"input_lammps":upload_artifact(tasks)},with_param=argo_range(n_model),key="MD-LAMMPS-{{item}}",util_image="registry.dp.tech/dptech/deepmd-kit:2.1.5-cuda11.6",util_command=['python3'])  
    wf.add(md_lammps)
    '''
    lammps_tem = PythonOPTemplate(lammps,slices=Slices("{{item}}", input_artifact=["input_lammps"],output_artifact=["output_lammps"]),image="registry.dp.tech/dptech/deepmd-kit:2.1.5-cuda11.6",command=["python3"])
    md_lammps = Step("MD-LAMMPS",template=lammps_tem,artifacts={"input_lammps":upload_artifact(tasks)},with_param=argo_range(n_model),key="MD-LAMMPS-{{item}}",util_image="registry.dp.tech/dptech/deepmd-kit:2.1.5-cuda11.6",util_command=['python3'],executor=BohriumExecutor(executor="bohrium_v2", extra={"scassType":"c32_m128_4 * NVIDIA V100","projectId": 10080,"jobType":"container", "logFiles": []}))   
    wf.add(md_lammps)
    
    ## Step2 prepare cp2k tasks
    dfttasks = Step(
        name="DFTTASKS", 
        template=PythonOPTemplate(DFTtasks,image="registry.dp.tech/dptech/prod-11461/calypso-need:calypso-need",command=["python3"]),
        artifacts={"input_dfttasks":md_lammps.outputs.artifacts["output_lammps"],
                   "cp2k_input":upload_artifact("template.inp")
                },
        parameters = {"nsample":nsample}
        )
    wf.add(dfttasks)

    ## Step3 run cp2k tasks
    cp2k = PythonOPTemplate(CP2K,slices=Slices("{{item}}", input_artifact=["input_cp2k"],output_artifact=["output_cp2k"]),image="registry.dp.tech/dptech/cp2k:7.1",command=["python3"],python_packages=["/usr/local/lib/python3.8/dist-packages/ase"])
    cp2k_cal = Step("CP2K-Cal",template=cp2k,artifacts={"input_cp2k":dfttasks.outputs.artifacts["output_dfttasks"]},with_param=argo_range(nsample),key="CP2K-Cal-{{item}}",util_image="registry.dp.tech/dptech/cp2k:7.1",util_command=['python3'],executor=BohriumExecutor(executor="bohrium_v2", extra={"scassType":"c32_m128_cpu","projectId": 10080,"jobType":"container", "logFiles": []}))   
    wf.add(cp2k_cal)

    ## Step4 collect *.cube files
    dftpost = Step(
        name="DFTPOST", 
        template=PythonOPTemplate(DFTpost,image="registry.dp.tech/dptech/prod-11461/chengqian:v1",command=["python3"]),
        artifacts={"input_dftpost":cp2k_cal.outputs.artifacts["output_cp2k"]},
        parameters = {"nsample":nsample}
        )
    wf.add(dftpost)

    wf.submit()

    while wf.query_status() in ["Pending","Running"]:
        time.sleep(4)
    assert(wf.query_status() == 'Succeeded')
    step1 = wf.query_step(name="MD-LAMMPS")[0]
    download_artifact(step1.outputs.artifacts["output_lammps"])


if __name__ == "__main__":
    brm_context = BohriumContext(
        executor="mixed",
        #executor="bohrium_v2",
        extra={"scass_type":"c32_m128_4 * NVIDIA V100","program_id":,"job_type":"container"}, # 全局bohrium配置
        username="",
        password=""
    ) 
    main()

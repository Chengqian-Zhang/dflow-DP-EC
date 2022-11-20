from dflow import config, s3_config
config["host"] = "https://workflow.dp.tech/"
s3_config["endpoint"] = "60.205.112.9:9000"
config["k8s_api_server"] = "https://182.92.168.135:6443"
config["token"] = "eyJhbGciOiJSUzI1NiIsImtpZCI6Im05a2kzYm1TUEhHVWxoeTk1ZVExNHBrZnBhM3FYbVFHb3VMU2d3NzM1NUEifQ.eyJpc3MiOiJrdWJlcm5ldGVzL3NlcnZpY2VhY2NvdW50Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9uYW1lc3BhY2UiOiJhcmdvIiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9zZWNyZXQubmFtZSI6ImFyZ28tdG9rZW4tcnZ2d2QiLCJrdWJlcm5ldGVzLmlvL3NlcnZpY2VhY2NvdW50L3NlcnZpY2UtYWNjb3VudC5uYW1lIjoiYXJnbyIsImt1YmVybmV0ZXMuaW8vc2VydmljZWFjY291bnQvc2VydmljZS1hY2NvdW50LnVpZCI6IjgwN2I4NzMzLTQwMTAtNDc0NC1hNDQxLTkzZGNjMTgwNzQyZCIsInN1YiI6InN5c3RlbTpzZXJ2aWNlYWNjb3VudDphcmdvOmFyZ28ifQ.NwpdoIBVg9GV2nQS24Cxrotg__MXlSsngjmpy7wcLhlHaHIwJYQFqCstDr2fpakzJOEAPxdn87SR4rjSSipecftUgE0IyBlKFcFx-sRlO0KHfFOQ83fcGJXe28N5qD1CWzgQqV9-s-WVWGBsmHcN2O4kxjfcfWUsOFspcoUFIw1ihuxGl5dj1fTJ6qtezChljB_gzGvUlwoJ_8oex3iT-BT_o9dJpBOTMtK0y93AJt4udLVbLUsYSjmQDrckx5nlHT6v1bQ-4K_8DAWNYXvzPwkGzTPhpO_zt-osBsvbjwAJViCbzklN27yU9eO0xfasC_8meefNt-LLq1wS59YeLg"

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

import subprocess, os, shutil, glob
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

def main():
    ## Organizing lammps input files
    n_model = 4
    cwd = os.getcwd()
    input_path = os.path.join(cwd,"01.dpmd","00.inputs")
    pot_path = os.path.join(cwd,"01.dpmd","01.pot")
    output_path = os.path.join(cwd,"01.dpmd","02.outputs")
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
    lammps_tem = PythonOPTemplate(lammps,slices=Slices("{{item}}", input_artifact=["input_lammps"],output_artifact=["output_lammps"]),image="registry.dp.tech/dptech/deepmd-kit:2.1.5-cuda11.6",command=["python3"])
    md_lammps = Step("MD-LAMMPS",template=lammps_tem,artifacts={"input_lammps":upload_artifact(tasks)},with_param=argo_range(n_model),key="MD-LAMMPS-{{item}}",util_image="registry.dp.tech/dptech/deepmd-kit:2.1.5-cuda11.6",util_command=['python3'])  
    wf.add(md_lammps)
    '''
    lammps_tem = PythonOPTemplate(lammps,slices=Slices("{{item}}", input_artifact=["input_lammps"],output_artifact=["output_lammps"]),image="registry.dp.tech/dptech/deepmd-kit:2.1.5-cuda11.6",command=["python3"])
    md_lammps = Step("MD-LAMMPS",template=lammps_tem,artifacts={"input_lammps":upload_artifact(tasks)},with_param=argo_range(n_model),key="MD-LAMMPS-{{item}}",util_image="registry.dp.tech/dptech/deepmd-kit:2.1.5-cuda11.6",util_command=['python3'],executor=BohriumExecutor(executor="bohrium_v2", extra={"scassType":"c32_m128_4 * NVIDIA V100","projectId": 10080,"jobType":"container", "logFiles": []}))   
    wf.add(md_lammps)
    
    ## Step2 
    
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
        extra={"scass_type":"c32_m128_4 * NVIDIA V100","program_id":10080,"job_type":"container"}, # 全局bohrium配置
        username="",
        password=""
    ) 
    main()

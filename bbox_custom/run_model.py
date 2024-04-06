import vart
import numpy as np
import torch
from vart import Runner
import xir
import torch.nn.functional as F

def get_child_subgraph_dpu(graph):
    assert graph is not None, "'graph' should not be None"
    root_subgraph = graph.get_root_subgraph()
    assert (root_subgraph is not None), "Failed to get root graph"
    print(root_subgraph, root_subgraph.is_leaf)
    if root_subgraph.is_leaf:
        return []

    child_subgraphs = root_subgraph.toposort_child_subgraph()
    return [
            cs
            for cs in child_subgraphs
            if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
            ]

def execute_async(dpu, tensor_buffers_dict):
    print("OUTPUT: ", dpu.get_output_tensors())
    input_tensor_buffers = [
    tensor_buffers_dict[t.name] for t in dpu.get_input_tensors()
            ]
    output_tensor_buffers = [
                    tensor_buffers_dict[t.name] for t in dpu.get_output_tensors()
                        ]
    jid = dpu.execute_async(input_tensor_buffers, output_tensor_buffers)
    return dpu.wait(jid)

def run_dpu(dpu):
    num_samples = 4
    num_channels = 3
    image_height = 128
    image_width = 128
    input_shape = (num_channels, image_height, image_width)

    input_tensor = dpu.get_input_tensors()
    output_tensor = dpu.get_output_tensors()
    print(output_tensor)

    input_dims = tuple(input_tensor[0].dims)
    output_dims_1 = tuple(output_tensor[0].dims)
    output_dims_2 = tuple(output_tensor[1].dims)
    print(output_dims_1, output_dims_2)
    out1 = np.zeros(output_dims_1, dtype='float32')
    out2 = np.zeros(output_dims_2, dtype='float32')
    #print(input_tensor)
    #print(output_tensor)
    batch_size = input_dims[0]
    print(batch_size)
    #print(batch_size)
    num_samples = 4
    output_shape = 1
    sample_input = torch.randn(num_samples, *input_shape)
    count = 0
    while count < num_samples:
        if (count + batch_size <= num_samples):
            runSize = batch_size 
        else:
            runSize = num_samples - count
        outputData = []
        inputData = []
        inputData = [np.empty(input_dims, dtype=np.float32, order="C")]

        for j in range(runSize):
            imageRun = inputData[0]
            random_data = np.random.rand(*input_dims[1:])
            #imageRun[j, :] = random_data

            inputData[0][j, ...] = random_data

        #execute_async(dpu, {"JacksCNNModel__input_0_fix": inputData[0], "JacksCNNModel__JacksCNNModel_ret_fix": out})
        print(inputData[0])
        execute_async(dpu, {"AirplaneDetector__input_0_fix": inputData[0], 
            "AirplaneDetector__AirplaneDetector_Linear_fc_bb__ret_fix": out1, 
            "AirplaneDetector__AirplaneDetector_Linear_fc_cls__ret_19_fix": out2})

        count += runSize 
        for i in range(runSize):
            print("first", out1[i])
            print("second", out2[i])
        #print(np.mean(out))
        #print(np.max(out), "max")

        #print(out.shape)


def main():
    # create graph runner
    xmodel_file = "bboxv1/bboxv1.xmodel"
    graph = xir.Graph.deserialize(xmodel_file)
    subgraph = graph.get_root_subgraph().toposort_child_subgraph()
    index = -1
    for i, s in enumerate(subgraph):
        print(s.get_attr("device"))
        if s.get_attr("device") == "DPU":
            index = i
    #print(subgraph)
    #print("here", dir(graph))
    #print(dir(Runner))
    runner = Runner.create_runner(subgraph[index], "run")
    run_dpu(runner)

    return

if __name__ == "__main__":
    main()
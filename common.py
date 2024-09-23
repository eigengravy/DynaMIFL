
def federated_averaging(global_model: nn.Module, models: List[nn.Module]) -> nn.Module:
    global_model = SimpleCNN().to(device)
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack(
            [client_model.state_dict()[k].float() for client_model in models], 0
        ).mean(0)
    global_model.load_state_dict(global_dict)
    return global_model


def calculate_mi(model: nn.Module, teacher: nn.Module, dataloader: DataLoader):
    model.eval()
    teacher.eval()
    
    all_model_outputs = []
    all_teacher_outputs = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0]  # Assuming the first element of the batch is the input
            inputs = inputs.to(device) 
            model_output = model(inputs).cpu().numpy()
            teacher_output = teacher(inputs).cpu().numpy()
            
            all_model_outputs.append(model_output)
            all_teacher_outputs.append(teacher_output)
    
    # Concatenate all outputs
    all_model_outputs = np.concatenate(all_model_outputs, axis=0)
    all_teacher_outputs = np.concatenate(all_teacher_outputs, axis=0)
    
    # Flatten the outputs if they're multi-dimensional
    all_model_outputs = all_model_outputs.reshape(-1)
    all_teacher_outputs = all_teacher_outputs.reshape(-1)
    
    # Calculate Pearson correlation coefficient
    rho, _ = pearsonr(all_model_outputs, all_teacher_outputs)
    
    # Calculate Mutual Information
    mi = -0.5 * math.log(1 - rho**2)
    
    return rho, mi
import torch

def analysis_masks(mask_dict):# retrun a string! please print it
    res=""
    # total information
    total=0
    non_zeros=0
    res_list=[]
    for key in mask_dict.keys():
        now_total=mask_dict[key].numel()
        total+=now_total
        now_non_zeros=mask_dict[key].sum()
        if now_non_zeros==0:
            print(key)
        non_zeros+=now_non_zeros
        kernel_size=mask_dict[key].shape
        res+="{} \t size {} total {} \t nonzero {} \t non_zero_rate {} \n".format(key,kernel_size,now_total,now_non_zeros,1-float(now_non_zeros)/now_total)
        res_list.append(1-float(now_non_zeros)/now_total)
    sparse_rate=1-(non_zeros/float(total))
    res="Total Sparse rate {} \n".format(sparse_rate)+res
    return res,res_list

def analysis_masks_with_writer(mask_dict,writer,epoch):# retrun a string! please print it
    res=""
    # total information
    total=0
    non_zeros=0
    res_list=[]
    connect=1
    for key in mask_dict.keys():
        now_total=mask_dict[key].numel()
        total+=now_total
        now_non_zeros=mask_dict[key].sum()
        if now_non_zeros==0:
            connect=0
        non_zeros+=now_non_zeros
        kernel_size=mask_dict[key].shape
        writer.add_scalar("Z_sparsy_{}".format(key),float(now_non_zeros)/float(now_total),global_step=epoch)
        res+="{} \t size {} total {} \t nonzero {} \t non_zero_rate {} \n".format(key,kernel_size,now_total,now_non_zeros,1-float(now_non_zeros)/now_total)
        res_list.append(1-float(now_non_zeros)/now_total)
    sparse_rate=(non_zeros/float(total))
    res="Total Sparse rate {} \n".format(sparse_rate)+res
    writer.add_scalar("Z_sparsy_total",sparse_rate,global_step=epoch)
    writer.add_scalar("Connect",connect,global_step=epoch)
    return res,res_list

def analysis_masks_plot(mask_dict):# retrun a string! please print it
    res=""
    # total information
    total=0
    non_zeros=0
    res_list=[]
    name_list=[]
    for key in mask_dict.keys():
        name_list.append(key)
        now_total=mask_dict[key].numel()
        total+=now_total
        now_non_zeros=mask_dict[key].sum()
        if now_non_zeros==0:
            print(key)
        non_zeros+=now_non_zeros
        kernel_size=mask_dict[key].shape
        res+="{} \t size {} total {} \t nonzero {} \t non_zero_rate {} \n".format(key,kernel_size,now_total,now_non_zeros,1-float(now_non_zeros)/now_total)
        res_list.append(1-float(now_non_zeros)/now_total)
    sparse_rate=1-(non_zeros/float(total))
    res="Total Sparse rate {} \n".format(sparse_rate)+res
    return res,res_list,name_list
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def compute_iou(box1, box2):
    """Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
    Return:
      (tensor) iou, sized [N,M].
    """
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(  # left top?  # torch.max(tensor1, tensor2) gives element-wise maximum
        box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    rb = torch.min(  # right bottom?
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    wh = rb - lt  # [N,M,2]
    wh[wh < 0] = 0  # clip at 0
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
    area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
    area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

    iou = inter / (area1 + area2 - inter)
    return iou


class YoloLoss(nn.Module):
    def __init__(self, S, B, l_coord, l_noobj):
        super(YoloLoss, self).__init__()
        self.S = S  # S x S is the number of grid cells
        self.B = B  # B is no of bboxes per grid cell
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def xywh2xyxy(self, boxes):
        """
        Parameters:
        boxes: (N,4) representing by x,y,w,h

        Returns:
        boxes: (N,4) representing by x1,y1,x2,y2

        if for a Box b the coordinates are represented by [x, y, w, h] then
        x1, y1 = x/S - 0.5*w, y/S - 0.5*h ; x2,y2 = x/S + 0.5*w, y/S + 0.5*h
        Note: Over here initially x, y are the center of the box and w,h are width and height.
        """
        ### CODE ###
        # Your code here
        boxes[:, 0] = boxes[:, 0] - 0.5 * boxes[:, 2]
        boxes[:, 1] = boxes[:, 1] - 0.5 * boxes[:, 3]
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        return boxes

    def find_best_iou_boxes(self, pred_box_list, box_target):
        """
        Parameters:
        box_pred_list : [(tensor) size (-1, 4) ...]  # this should be (-1, 5) !!! can't filter conf values otherwise!
        box_target : (tensor)  size (-1, 5)

        Returns:
        best_iou: (tensor) size (-1, 1)
        best_boxes : (tensor) size (-1, 5), containing the boxes which give the best iou among the two (self.B) predictions

        Hints:
        1) Find the iou's of each of the 2 bounding boxes of each grid cell of each image.
        2) For finding iou's use the compute_iou function
        3) use xywh2xyxy to convert bbox format if necessary,
        Note: Over here initially x, y are the center of the box and w,h are width and height.
        We perform this transformation to convert the correct coordinates into bounding box coordinates.
        """

        ### CODE ###
        # Your code here

        # Step 1: xywh2xyxy
        # print("No of items in pred_box_list: ", len(pred_box_list))
        # print("No of items in 1st item of pred_box_list: ", pred_box_list[0].shape)
        # print("No of items in 2nd item of pred_box_list: ", pred_box_list[1].shape)

        ious = []
        # best_boxes = pred_box_list[0]  # init
        for box in pred_box_list:
            # print("should be (-1,5)", box.shape)
            iou = torch.diagonal(compute_iou(box[:, 0:4], box_target[:, 0:4]), offset=0)  # IOU of a bbox from the list of the 'i'th grid cell with the 'i'th target
            # now iou is a (-1) sized 1-D tensor
            iou = torch.unsqueeze(iou, dim=1)
            # now iou is a (-1, 1) sized 1-D tensor
            ious.append(iou)
        #ious_tensor = torch.Tensor(ious[0].shape[0], self.B)  # https://discuss.pytorch.org/t/how-to-turn-a-list-of-tensor-to-tensor/8868/3
        ious_tensor = torch.cat(ious, dim=1)
        best_iou, indices = torch.max(ious_tensor, dim=1)  # max along dim 1 reduces the Self.B many ious into a single best iou.

        best_boxes_stack = torch.stack(pred_box_list)[indices]
        best_boxes = best_boxes_stack[torch.arange(indices.shape[0]), torch.arange(indices.shape[0]), :]  # shape (-1, 1, 5)
        best_boxes = torch.squeeze(best_boxes, dim=1)  # should give shape (-1,5)

        return best_iou, best_boxes

    def get_class_prediction_loss(self, classes_pred, classes_target, has_object_map):
        """
        Parameters:
        classes_pred : (tensor) size (batch_size, S, S, 20)
        classes_target : (tensor) size (batch_size, S, S, 20)
        has_object_map: (tensor) size (batch_size, S, S)

        Returns:
        class_loss : scalar
        """
        ### CODE ###
        # Your code here
        error = classes_target - classes_pred
        error = (error * error) * has_object_map.unsqueeze(-1).expand_as(error)  # element-wise
        loss = torch.sum(error)  # scalar in a tensor -- imp for gradients to pass through during backprop

        return loss

    def get_no_object_loss(self, pred_boxes_list, has_object_map):
        """
        Parameters:
        pred_boxes_list: (list) [(tensor) size (N, S, S, 5)  for B pred_boxes]
        has_object_map: (tensor) size (N, S, S)

        Returns:
        loss : scalar

        Hints:
        1) Only compute loss for cell which doesn't contain object
        2) compute loss for all predictions in the pred_boxes_list list
        3) You can assume the ground truth confidence of non-object cells is 0
        """
        ### CODE ###
        # Your code here
        C_GT = torch.zeros(has_object_map.shape, device=has_object_map.get_device())  # shape (N,S,S)
        # temp = torch.unsqueeze(torch.zeros_like(has_object_map), dim=3)
        # C_GT = torch.cat((temp, temp), -1)    # -1 means cat along last dim
        
        """ commented code doesn't work as intended
        boxes_conf_list = [tensor[:, :, :, 4] for tensor in pred_boxes_list]  # list of tensors of shape (N,S,S,1)
        # ^^conf values assumed to be in 4th index of 4th dim
        pred_boxes_conf_unsqueezed = torch.stack(boxes_conf_list, dim=0)  # shape (B,N,S,S,1)
        print(pred_boxes_conf_unsqueezed.shape)
        pred_boxes_conf = torch.squeeze(pred_boxes_conf_unsqueezed, dim=4)  # shape (B,N,S,S)
        """
        pred_boxes_list_tensor = torch.stack(pred_boxes_list, dim=0)  # shape (B,N,S,S,5)
        pred_boxes_conf = pred_boxes_list_tensor[:,:,:,:,4] # shape (B,N,S,S)
        
        error = pred_boxes_conf - C_GT.expand(self.B, -1, -1, -1)  # expand can add dimension only in front
        no_object_map = torch.ones(has_object_map.shape, device=has_object_map.get_device()) - (torch.ones(has_object_map.shape, device=has_object_map.get_device())*has_object_map) # converts [True, False] to [0, 1]
        error = (error * error) * no_object_map.expand(self.B, -1, -1, -1)  # element-wise mul
        loss = torch.sum(error) * self.l_noobj  # scalar in a tensor -- imp for gradients to pass during backprop

        return loss

    def get_contain_conf_loss(self, box_pred_conf, box_target_conf):
        """
        Parameters:
        box_pred_conf : (tensor) size (-1,1)
        box_target_conf: (tensor) size (-1,1)

        Returns:
        contain_loss : scalar

        Hints:
        The box_target_conf should be treated as ground truth, i.e., no gradient

        """
        ### CODE
        # your code here
        GT = box_target_conf.detach()
        # detach is necessary since we do not want bo_target_conf's gradients to be messed up/ mess other tensors up
        error = box_pred_conf - GT
        error = error * error
        loss = torch.sum(error) * self.l_coord   # scalar in a tensor -- imp for gradients to pass during backprop

        return loss

    def get_regression_loss(self, box_pred_response, box_target_response):
        """
        Parameters:
        box_pred_response : (tensor) size (-1, 4)
        box_target_response : (tensor) size (-1, 4)
        Note : -1 corresponds to ravels the tensor into the dimension specified
        See : https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view_as

        Returns:
        reg_loss : scalar

        """
        ### CODE
        # your code here
        error_x = box_pred_response[:, 0] - box_target_response[:, 0]
        error_y = box_pred_response[:, 1] - box_target_response[:, 1]
        error_w = (box_pred_response[:, 2])**0.5 - (box_target_response[:, 2])**0.5
        error_h = (box_pred_response[:, 3])**0.5 - (box_target_response[:, 3])**0.5

        loss_coord = torch.sum((error_x*error_x) + (error_y*error_y))
        loss_size = torch.sum((error_w*error_w) + (error_h*error_h))
        reg_loss = self.l_coord * (loss_size + loss_coord)   # scalar in a tensor -- imp for gradients to pass during backprop

        return reg_loss

    def forward(self, pred_tensor, target_boxes, target_cls, has_object_map):
        """
        pred_tensor: (tensor) size(N,S,S,Bx5+20=30) N:batch_size
                      where B - number of bounding boxes this grid cell is a part of = 2
                            5 - number of bounding box values corresponding to [x, y, w, h, c]
                                where x - x_coord, y - y_coord, w - width, h - height, c - confidence of having an object
                            20 - number of classes

        target_boxes: (tensor) size (N, S, S, 4): the ground truth bounding boxes
        target_cls: (tensor) size (N, S, S, 20): the ground truth class
        has_object_map: (tensor, bool) size (N, S, S): the ground truth for whether each cell contains an object (True/False)

        Returns:
        loss_dict (dict): with key value stored for total_loss, reg_loss, containing_obj_loss, no_obj_loss and cls_loss
        """
        N = pred_tensor.size(0)
        total_loss = 0.0

        # split the pred tensor from an entity to separate tensors:
        # -- pred_boxes_list: a list containing all bbox prediction (list) [(tensor) size (N, S, S, 5)  for B pred_boxes]
        # -- pred_cls (containing all classification prediction)
        pred_boxes_tensor = pred_tensor[:, :, :, 0:(self.B*5)]
        pred_boxes_list = []
        for i in range(0, (self.B*5), 5):
            pred_boxes_list.append(pred_boxes_tensor[:, :, :, i:(i+5)])  # list of tensors of shape (N,S,S,5)
        pred_cls = pred_tensor[:, :, :, (self.B*5):pred_tensor.size(3)]  # shape (N,S,S,20)

        # compute classification loss
        cls_loss = (self.get_class_prediction_loss(pred_cls, target_cls, has_object_map))/N

        # compute no-object loss
        no_obj_loss = (self.get_no_object_loss(pred_boxes_list, has_object_map))/N

        # Re-shape boxes in pred_boxes_list and target_boxes to meet the following desires
        # 1) only keep having-object cells
        # 2) vectorize all dimensions except for the last one for faster computation

        for i in range(len(pred_boxes_list)):
            # filter out the cells where obj doesn't exist
            box = pred_boxes_list[i]
            box = box[has_object_map.unsqueeze(-1).expand_as(box) == True]
            # shape (something less than N*S*S, 5)
            pred_boxes_list[i] = box.reshape((-1, 5)) 
            #print("should be (-1, 5)", box.shape)
        #print(pred_boxes_list[0].shape)

        target_boxes = target_boxes[has_object_map.unsqueeze(-1).expand_as(target_boxes) == True]
        target_boxes = target_boxes.reshape((-1, 4))

        # find the best boxes among the 2 (or self.B) predicted boxes and the corresponding iou
        best_iou, best_boxes = self.find_best_iou_boxes(pred_boxes_list, target_boxes)

        # compute regression loss between the found best bbox and GT bbox for all the cell containing objects
        reg_loss = (self.get_regression_loss(best_boxes[:, 0:4], target_boxes))/N  # sending only pred_best_boxes & GT bboxes where object exists

        # compute contain_object_loss
        containing_obj_loss = (self.get_contain_conf_loss(best_boxes[:, 4], best_iou))/N

        # compute final loss
        total_loss = reg_loss + containing_obj_loss + no_obj_loss + cls_loss

        # construct return loss_dict
        loss_dict = dict(
            total_loss=total_loss,
            reg_loss=reg_loss,
            containing_obj_loss=containing_obj_loss,
            no_obj_loss=no_obj_loss,
            cls_loss=cls_loss
        )
        return loss_dict
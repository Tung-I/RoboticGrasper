import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
import random
import os
import sys
from gym import spaces
import time
import pybullet as p
import numpy as np
import pybullet_data
import glob
from pkg_resources import parse_version
import gym
from bullet.tm700 import tm700
from unused_code.tm700_possensorbothgrippers_Gym import tm700_possensor_gym
from pathlib import Path
from tqdm import tqdm

IM_WIDTH = 320
IM_HEIGHT = 320
N_DATA_TO_GENERATE = 20
# N_UNSEEN_DATA_TO_GENERATE = 500
# DATA_ROOT = Path('/data/ShapeNet_subset/')
cwd = os.getcwd()
DATA_ROOT = Path(cwd + '/data/ShapeNet_subset')
OUTPUT_DIR = cwd + '/output'

# NAME2IDX = {
#     'airplane': 1, 'car': 2, 'guitar': 3, 'laptop': 4, 'pistol': 5, 'bag': 6, 'chair': 7, 'knife': 8, 'motorbike': 9, 'rocket': 10, 
#     'table': 11, 'cap': 12, 'earphone': 13, 'lamp': 14, 'mug': 15, 'skateboard': 16
# }
# UNSEEN = ['rocket', 'lamp', 'pistol', 'car']  # 21855, 6243

CLASS_NAME = ['airplane', 'car', 'chair', 'guitar', 'mug']
NAME2IDX = {'airplane': 1, 'car': 2, 'chair': 3, 'guitar': 4, 'mug': 5}

RESUME_NUM = None
# TASK_SEEN = False
# TASK_UNSEEN = True
USE_GUI = False

class tm700_rgbd_gym(tm700_possensor_gym):
    def __init__(self,
               urdfRoot=pybullet_data.getDataPath(),
               actionRepeat=80,
               isEnableSelfCollision=True,
               renders=False,
               isDiscrete=True,
               maxSteps=11,
               dv=0.06,
               removeHeightHack=False,
               blockRandom=0.1,
               cameraRandom=0,
               width=64,
               height=64,
               numObjects=5,
               isTest=False):

        ##############
        self._width = IM_WIDTH
        self._height = IM_HEIGHT
        self.img_save_cnt = 0
        self.N_DATA_TO_GENERATE = N_DATA_TO_GENERATE
        ##############

        self._isDiscrete = isDiscrete
        # self._timeStep = 1. / 240.
        self._timeStep = 1. / 60.
        self._urdfRoot = urdfRoot
        self._actionRepeat = actionRepeat
        self._isEnableSelfCollision = isEnableSelfCollision
        self._observation = []
        self._envStepCounter = 0
        self._renders = renders
        self._maxSteps = maxSteps
        self.terminated = 0
        self._cam_dist = 1.3
        self._cam_yaw = 180
        self._cam_pitch = -40
        self._dv = dv
        self._p = p
        self._removeHeightHack = removeHeightHack
        self._blockRandom = blockRandom
        self._cameraRandom = cameraRandom
        self._isTest = isTest
        self.observation_space = spaces.Box(low=0,
                                            high=255,
                                            shape=(self._height, self._width, 3),
                                            dtype=np.uint8)
        #############################
        self.model_paths = self.get_data_path()
        #############################
        # disable GUI or not
        if USE_GUI:
            self.cid = p.connect(p.SHARED_MEMORY)
            if (self.cid < 0):
                self.cid = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, -0.2, -0.33]) # cameraposition of rendering
        else:
            self.cid = p.connect(p.DIRECT)

        self.seed()

        if (self._isDiscrete):
            if self._removeHeightHack:
                self.action_space = spaces.Discrete(9)
            else:
                self.action_space = spaces.Discrete(7)
        else:
            self.action_space = spaces.Box(low=-1, high=1, shape=(3,))  # dx, dy, da
            if self._removeHeightHack:
                self.action_space = spaces.Box(low=-1, high=1, shape=(4,))  # dx, dy, dz, da
        self.viewer = None


    def get_data_path(self):
        # print('#######################')
        # print(c_d.parts[-1])
        # raise Exception(' ')

        root = DATA_ROOT
        class_dirs = [c for c in root.iterdir() if c.is_dir() and 'ipynb' not in str(c)]
        obj_pool = {}
        for c_n in CLASS_NAME:
            obj_pool[c_n] = []

        for c_d in class_dirs:
            current_class_name = c_d.parts[-1]
            current_class_id = NAME2IDX[current_class_name]
            obj_dirs = [o for o in c_d.iterdir() if o.is_dir()]
            for _dir in obj_dirs:
                obj_path = str(_dir / Path('models/model_normalized.obj'))
                obj_pool[current_class_name].append(obj_path)
            
        random.seed(0)
        for c_n in CLASS_NAME:
            random.shuffle(obj_pool[c_n])
        return obj_pool


    def reset(self):
        """Environment reset called at the beginning of an episode.
        """
        look = [0.4, 0.1, 0.54] 
        distance = 1.5
        pitch = -90
        yaw = -90
        roll = 180
        pos_range = [0.45, 0.5, 0.0, 0.1]

        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(look, distance, yaw, pitch, roll, 2)
        fov = 20. + self._cameraRandom * np.random.uniform(-2, 2)
        aspect = self._width / self._height
        near = 0.01
        far = 10
        self._proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

        self._attempted_grasp = False
        self._env_step = 0
        self.terminated = 0

        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timeStep)
        p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), [0, 0, -1])

        self.tableUid = p.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"), 0.5000000, 0.00000, -.640000,
                                0.000000, 0.000000, 0.0, 1.0)

        p.setGravity(0, 0, -10)
        self._envStepCounter = 0
        p.stepSimulation()

        # place objs
        if RESUME_NUM is not None:
            self.model_paths = self.model_paths[RESUME_NUM*3:]
            self.img_save_cnt = RESUME_NUM
            self.N_DATA_TO_GENERATE -= RESUME_NUM
        for _ in tqdm(range(self.N_DATA_TO_GENERATE)):
            urdfList = []
            chosen_classes = random.sample(CLASS_NAME, k=3)
            for _c in chosen_classes:
                urdfList.append([self.model_paths[_c].pop(), _c])
            self._objectUids, self._objectClasses = self._randomly_place_objects(urdfList, pos_range)
            self._observation = self._get_observation()
            self.img_save_cnt += 1
            for uid in self._objectUids:
                p.removeBody(uid)

            # close-up view
            for idx, obj_path in enumerate(urdfList):
                self._objectUids, self._objectClasses = self._place_single_object([obj_path])
                observation = self._get_closeup(idx)
                for uid in self._objectUids:
                    p.removeBody(uid)


        # terminate
        sys.exit('finish')

        return np.array(self._observation)


    def _randomly_place_objects(self, urdfList, pos_range):
        objectUids = []
        objectClasses = []
        shift = [0, -0.02, 0]
        meshScale = [0.3, 0.3, 0.3]
        list_x_pos = []
        list_y_pos = []
        random.seed(self.img_save_cnt)

        # for idx, (_, _) in enumerate(urdfList):
        #     list_x_pos.append(random.uniform(pos_range[0], pos_range[1]))
        #     list_y_pos.append(random.uniform(pos_range[2], pos_range[3]))
        # pos_range = [0.45, 0.5, 0.0, 0.1]
        list_x_pos = [0.45, 0.5, 0.55]
        list_y_pos = [0.15, 0.05, 0.15]

        for idx, (urdf_path, class_name) in enumerate(urdfList):
            # print(urdf_path)
            visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,
                                                fileName=urdf_path,
                                                rgbaColor=[1, 1, 1, 1],
                                                specularColor=[0.4, .4, 0],
                                                visualFramePosition=shift,
                                                meshScale=meshScale)
            try:
                collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                                    fileName=urdf_path,
                                                    collisionFramePosition=shift,
                                                    meshScale=meshScale)
            except:
                print(urdf_path)
                continue

            xpos = list_x_pos[idx]
            ypos = list_y_pos[idx]

            uid = p.createMultiBody(baseMass=1,
                            baseInertialFramePosition=[-0.2, 0, 0],
                            baseCollisionShapeIndex=collisionShapeId,
                            baseVisualShapeIndex=visualShapeId,
                            basePosition=[xpos, ypos, .15],
                            useMaximalCoordinates=True)

            objectUids.append(uid)
            objectClasses.append(class_name)
            for _ in range(150):
                p.stepSimulation()

        return objectUids, objectClasses


    def _place_single_object(self, urdfList):
        objectUids = []
        objectClasses = []
        shift = [0, -0.02, 0]
        # meshScale = [0.3, 0.3, 0.3]
        meshScale = [0.6, 0.6, 0.6]
        list_x_pos = []
        list_y_pos = []
        xpos = 0.5
        ypos = 0.15

        for idx, (urdf_path, class_name) in enumerate(urdfList):
            # print(urdf_path)
            visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,
                                                fileName=urdf_path,
                                                rgbaColor=[1, 1, 1, 1],
                                                specularColor=[0.4, .4, 0],
                                                visualFramePosition=shift,
                                                meshScale=meshScale)
            try:
                collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                                    fileName=urdf_path,
                                                    collisionFramePosition=shift,
                                                    meshScale=meshScale)
            except:
                print(urdf_path)
                continue

        uid = p.createMultiBody(baseMass=1,
                        baseInertialFramePosition=[-0.2, 0, 0],
                        baseCollisionShapeIndex=collisionShapeId,
                        baseVisualShapeIndex=visualShapeId,
                        basePosition=[xpos, ypos, .11],
                        useMaximalCoordinates=True)

        objectUids.append(uid)
        objectClasses.append(class_name)
        for _ in range(150):
            p.stepSimulation()
        return objectUids, objectClasses


    def _get_closeup(self, idx):
        """Return the observation as an image.
        """
        img_arr = p.getCameraImage(width=self._width,
                                height=self._height,
                                viewMatrix=self._view_matrix,
                                projectionMatrix=self._proj_matrix)
        rgb = img_arr[2]
        depth = img_arr[3]
        min = 0.97
        max=1.0
        segmentation = img_arr[4]
        depth = np.reshape(depth, (self._height, self._width,1) )
        segmentation = np.reshape(segmentation, (self._height, self._width,1) )

        np_img_arr = np.reshape(rgb, (self._height, self._width, 4))
        np_img_arr = np_img_arr[:, :, :3].astype(np.float64)

        view_mat = np.asarray(self._view_matrix).reshape(4, 4)
        proj_mat = np.asarray(self._proj_matrix).reshape(4, 4)
        # pos = np.reshape(np.asarray(list(p.getBasePositionAndOrientation(self._objectUids[0])[0])+[1]), (4, 1))

        AABBs = np.zeros((len(self._objectUids), 2, 3))
        cls_ls = []
        
        for i, (_uid, _cls) in enumerate(zip(self._objectUids, self._objectClasses)):
            AABBs[i] = np.asarray(p.getAABB(_uid)).reshape(2, 3)
            cls_ls.append(NAME2IDX[_cls])

        # np.save('/home/tony/Desktop/obj_save/view_mat_'+str(self.img_save_cnt), view_mat)
        # np.save('/home/tony/Desktop/obj_save/proj_mat_'+str(self.img_save_cnt), proj_mat)
        # np.save('/home/tony/Desktop/obj_save/img_'+str(self.img_save_cnt), np_img_arr.astype(np.int16))
        # np.save('/home/tony/Desktop/obj_save/AABB_'+str(self.img_save_cnt), AABBs)
        # np.save('/home/tony/Desktop/obj_save/class_'+str(self.img_save_cnt), np.array(cls_ls))

        np.save(OUTPUT_DIR + '/closeup_' + str(self.img_save_cnt - 1) + '_' + str(idx), np_img_arr.astype(np.int16))
        dets = np.zeros((AABBs.shape[0], 5))
        for i in range(AABBs.shape[0]):
            dets[i, :4] = self.get_2d_bbox(AABBs[i], view_mat, proj_mat, IM_HEIGHT, IM_WIDTH)
            dets[i, 4] = int(cls_ls[i])
        # np.save(OUTPUT_DIR + '/annotation_'+str(self.img_save_cnt), dets)

        test = np.concatenate([np_img_arr[:, :, 0:2], segmentation], axis=-1)

        return test


    def _get_observation(self, unseen=False):
        """Return the observation as an image.
        """
        img_arr = p.getCameraImage(width=self._width,
                                height=self._height,
                                viewMatrix=self._view_matrix,
                                projectionMatrix=self._proj_matrix)
        rgb = img_arr[2]
        depth = img_arr[3]
        min = 0.97
        max=1.0
        segmentation = img_arr[4]
        depth = np.reshape(depth, (self._height, self._width,1) )
        segmentation = np.reshape(segmentation, (self._height, self._width,1) )

        np_img_arr = np.reshape(rgb, (self._height, self._width, 4))
        np_img_arr = np_img_arr[:, :, :3].astype(np.float64)

        view_mat = np.asarray(self._view_matrix).reshape(4, 4)
        proj_mat = np.asarray(self._proj_matrix).reshape(4, 4)
        # pos = np.reshape(np.asarray(list(p.getBasePositionAndOrientation(self._objectUids[0])[0])+[1]), (4, 1))

        AABBs = np.zeros((len(self._objectUids), 2, 3))
        cls_ls = []
        
        for i, (_uid, _cls) in enumerate(zip(self._objectUids, self._objectClasses)):
            AABBs[i] = np.asarray(p.getAABB(_uid)).reshape(2, 3)
            cls_ls.append(NAME2IDX[_cls])

        # np.save('/home/tony/Desktop/obj_save/view_mat_'+str(self.img_save_cnt), view_mat)
        # np.save('/home/tony/Desktop/obj_save/proj_mat_'+str(self.img_save_cnt), proj_mat)
        # np.save('/home/tony/Desktop/obj_save/img_'+str(self.img_save_cnt), np_img_arr.astype(np.int16))
        # np.save('/home/tony/Desktop/obj_save/AABB_'+str(self.img_save_cnt), AABBs)
        # np.save('/home/tony/Desktop/obj_save/class_'+str(self.img_save_cnt), np.array(cls_ls))

        np.save(OUTPUT_DIR + '/image_' + str(self.img_save_cnt), np_img_arr.astype(np.int16))
        dets = np.zeros((AABBs.shape[0], 5))
        for i in range(AABBs.shape[0]):
            dets[i, :4] = self.get_2d_bbox(AABBs[i], view_mat, proj_mat, IM_HEIGHT, IM_WIDTH)
            dets[i, 4] = int(cls_ls[i])
        np.save(OUTPUT_DIR + '/annotation_'+str(self.img_save_cnt), dets)

        test = np.concatenate([np_img_arr[:, :, 0:2], segmentation], axis=-1)

        return test


    def get_img_xy(self, xyz, mat_view, mat_proj, pixel_h, pixel_w):
        xyz = np.concatenate([xyz, np.asarray([1.])], axis=0)
        mat_view = np.asarray(mat_view).reshape(4, 4)
        mat_proj = np.asarray(mat_proj).reshape(4, 4)
        xyz = np.dot(xyz, mat_view)
        xyz = np.dot(xyz, mat_proj)
        u, v, z = xyz[:3]
        u = u  / z  * (pixel_w / 2) + (pixel_w / 2)
        v = (1 - v / z)  *  pixel_h / 2
        return u, v

    def get_2d_bbox(self, aabb, mat_view, mat_proj, pixel_h, pixel_w):
        x_min, y_min, z_min = aabb[0]
        x_max, y_max, z_max = aabb[1]
        top = float('inf')
        bot = 0
        left = float('inf')
        right = 0
        for _x in [x_min, x_max]:
            for _y in [y_min, y_max]:
                for _z in [z_min, z_max]:
                    xyz = [_x, _y, _z]
                    u, v = self.get_img_xy(xyz, mat_view, mat_proj, pixel_h, pixel_w)
                    if u > right:
                        right = u if u < pixel_w else pixel_w - 1
                    if u < left:
                        left = u if u >=0 else 0
                    if v > bot:
                        bot = v if v < pixel_h else pixel_h - 1
                    if v < top:
                        top = v if v >= 0 else 0
        w_bbox = right - left
        h_bbox = bot - top
        return left, top, right, bot

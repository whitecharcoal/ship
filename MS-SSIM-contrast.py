import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# 参数设置
DATABASE_PATH = r"D:\TaoWeiHao\ALBK_Template\ALBK_25-35_0.1"  # 模板库路径
QUERY_IMAGE_PATH = r"D:\TaoWeiHao\SSIM\test\30.0_noise0.30.jpg"  # 待查询图像路径
NUM_SCALES = 5  # MS-SSIM尺度数
CONTRASTIVE_MARGIN = 0.15  # 对比学习边界值
LEARNING_RATE = 0.003  # 学习率
ITERATIONS = 500  # 对比学习迭代次数
init_weights = np.array([0.0448, 0.2856, 0.3000, 0.2363, 0.1333])

filename = os.path.basename(QUERY_IMAGE_PATH)
# true_angle = float(filename.split('_')[0])
true_angle = 25.0

ANGLE_START = 25.0  # 起始角度
ANGLE_END = 35.0  # 结束角度
ANGLE_STEP = 0.1  # 角度间隔


def generate_angles():
    return np.arange(ANGLE_START, ANGLE_END + ANGLE_STEP / 2, ANGLE_STEP).round(1)


def generate_valid_neg_angles(pos_angle, num_neg=2):
    """生成有效负样本角度列表"""
    all_angles = generate_angles()
    valid_neg = []

    # 双向相邻采样
    right_neg = round(pos_angle + ANGLE_STEP, 1)
    if right_neg in all_angles:
        valid_neg.append(right_neg)

    left_neg = round(pos_angle - ANGLE_STEP, 1)
    if left_neg in all_angles:
        valid_neg.append(left_neg)

    # 随机补充采样
    remaining_neg = [a for a in all_angles if a not in valid_neg + [pos_angle]]
    if len(valid_neg) < num_neg and remaining_neg:
        need = num_neg - len(valid_neg)
        valid_neg += list(np.random.choice(remaining_neg, size=need, replace=False))

    return valid_neg[:num_neg]  # 返回指定数量的负样本

def load_image(img_path):
    """直接加载原图并转为灰度"""
    img = cv2.imread(img_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def compute_ms_ssim(img1, img2, weights):
    """论文标准版MS-SSIM实现"""
    total_scales = len(weights)
    product = 1.0

    for s in range(len(weights)):
        scale = 2 ** s
        # 降采样
        img1_scaled = cv2.resize(img1, None, fx=1 / scale, fy=1 / scale,
                                 interpolation=cv2.INTER_AREA)
        img2_scaled = cv2.resize(img2, None, fx=1 / scale, fy=1 / scale,
                                 interpolation=cv2.INTER_AREA)


        # 单独计算亮度(l)、对比度(c)、结构(s)分量
        mu_x = np.mean(img1_scaled)
        mu_y = np.mean(img2_scaled)
        sigma_x = np.std(img1_scaled)
        sigma_y = np.std(img2_scaled)
        sigma_xy = np.cov(img1_scaled.flatten(), img2_scaled.flatten())[0, 1]

        # SSIM常数（防止除以零）
        C1 = (0.01 * 255) ** 2   # 亮度常数
        C2 = (0.03 * 255) ** 2   # 对比度/结构常数

        # 按尺度处理：最高层包含亮度，其他层仅对比度+结构
        if s == total_scales - 1:
            l = (2*mu_x*mu_y + C1) / (mu_x**2 + mu_y**2 + C1)
        else:
            l = 1.0  # 非最高层的亮度分量设为1（不影响乘积）

        c = (2 * sigma_x * sigma_y + C2) / (sigma_x ** 2 + sigma_y ** 2 + C2)
        ss = (sigma_xy + C2 / 2) / (sigma_x * sigma_y + C2 / 2)

        # 按权重组构各尺度项（注意指数与权重的对应关系）
        term = (l * c * ss) ** weights[s]
        product *= term

    # 各尺度结果相乘
    return product


def train_contrastive(anchor_img, pos_imgs, neg_imgs, init_weights, lr, iterations, margin):
    """对比学习优化权重（保持不变）"""
    weights = np.array(init_weights, dtype=np.float32)
    # 添加训练进度条
    with tqdm(total=iterations, desc="对比学习训练", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
        for iter in range(iterations):
            total_grad = np.zeros_like(weights)
            total_loss = 0.0

            for i in range(len(pos_imgs)):
                sim_pos = compute_ms_ssim(anchor_img, pos_imgs[i], weights)   # 正样本对相似度
                sim_neg = compute_ms_ssim(anchor_img, neg_imgs[i], weights)   # 负样本对相似度

                loss = max(sim_neg - sim_pos + margin, 0)
                total_loss += loss

                if loss > 0:
                    grad_pos = gradient_ms_ssim(anchor_img, pos_imgs[i], weights)
                    grad_neg = gradient_ms_ssim(anchor_img, neg_imgs[i], weights)
                    total_grad += (grad_neg - grad_pos)

            if len(pos_imgs) > 0:
                total_grad /= len(pos_imgs)   # 平均梯度（批次归一化）

            weights -= lr * total_grad

            # == 投影约束（权重可行域保障）==c
            weights = np.maximum(weights, 0)    # 非负约束
            weights /= (weights.sum() + 1e-8)   # 归一化（L1约束）

            # 每10次迭代输出详细信息
            if iter % 10 == 0:
                # 使用tqdm.write避免干扰进度条
                pbar.write("\n" + "=" * 40)
                pbar.write(f"迭代次数: {iter}")
                pbar.write("当前权重分布:")
                for scale, weight in enumerate(weights):
                    pbar.write(f"尺度 {scale + 1}: {weight:.4f}")
                pbar.write(f"当前损失: {total_loss:.4f}")
                pbar.write("=" * 40 + "\n")

                # 更新进度条状态
            pbar.set_postfix({
                'loss': f"{total_loss:.4f}",
                'weights': np.round(weights, 3)
            })
            pbar.update(1)

    return weights


def gradient_ms_ssim(img1, img2, weights):
    """基于MS-SSIM分量解析的梯度计算函数"""
    scales = len(weights)
    grads = np.zeros(scales)
    log_terms = []  # 存储各尺度的对数项（用于梯度运算）

    # === 前向过程（分解各分量） ===
    msssim = 1.0
    for s in range(scales):
        scale = 2 ** s
        img1_scale = cv2.resize(img1, None, fx=1 / scale, fy=1 / scale,
                                 interpolation=cv2.INTER_AREA)
        img2_scale = cv2.resize(img2, None, fx=1 / scale, fy=1 / scale,
                                 interpolation=cv2.INTER_AREA)

        mu_x = np.mean(img1_scale)
        mu_y = np.mean(img2_scale)
        sigma_x = np.std(img1_scale)
        sigma_y = np.std(img2_scale)
        sigma_xy = np.cov(img1_scale.flatten(), img2_scale.flatten())[0, 1]

        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        # 分量计算
        if s == scales - 1:
            l = (2 * mu_x * mu_y + C1) / (mu_x ** 2 + mu_y ** 2 + C1)
        else:
            l = 1.0

        c = (2 * sigma_x * sigma_y + C2) / (sigma_x ** 2 + sigma_y ** 2 + C2)
        ss = (sigma_xy + C2 / 2) / (sigma_x * sigma_y + C2 / 2)

        # ===== 关键修改：分尺度记录对数项 =====
        if s == scales - 1:  # 最高尺度（包含亮度）
            log_term = np.log(l + 1e-8) + np.log(c + 1e-8) + np.log(ss + 1e-8)
        else:  # 非最高尺度（只有对比度、结构）
            log_term = np.log(c + 1e-8) + np.log(ss + 1e-8)

        log_terms.append(log_term)
        msssim *= ((l * c * ss) ** weights[s])  # 注意：非最高层时l=1

        # Step 2: 反向传播梯度
    for s in range(scales):
        # 梯度公式：∂S/∂w_s = S * log_term[s]
        grads[s] = msssim * log_terms[s]

        # 数值稳定处理（防止权重趋于零时的溢出）
        if weights[s] > 1e-8:
            grads[s] /= weights[s]
        else:
            grads[s] = 0  # 权重接近于零时不更新该尺度

    return grads


def load_database_images(db_path):
    """加载数据库原图"""
    cache_path = "database_images.npy"
    if os.path.exists(cache_path):
        return np.load(cache_path, allow_pickle=True)

    angles = generate_angles()
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(load_image, os.path.join(db_path, f"{a:.1f}.png"))
                   for a in angles]
        images = [f.result() for f in tqdm(futures, desc="Loading Database")]

    np.save(cache_path, images)
    return images


if __name__ == "__main__":

    # 加载查询图像（添加进度提示）
    print("正在加载查询图像...")
    query_img = load_image(QUERY_IMAGE_PATH)

    # 对比学习优化权重
    if not os.path.exists("optimized_weights.npy"):
        print("初始化对比学习训练...")
        pos_angles = [true_angle]
        neg_angles = generate_valid_neg_angles(true_angle, num_neg=3)

        print("加载训练样本...")
        pos_imgs = [load_image(os.path.join(DATABASE_PATH, f"{a:.1f}.png"))
                    for a in tqdm(pos_angles, desc="正样本加载")]
        neg_imgs = [load_image(os.path.join(DATABASE_PATH, f"{a:.1f}.png"))
                    for a in tqdm(neg_angles, desc="负样本加载")]

        opt_weights = train_contrastive(query_img, pos_imgs, neg_imgs, init_weights,
                                        LEARNING_RATE, ITERATIONS, CONTRASTIVE_MARGIN)
        np.save("optimized_weights.npy", opt_weights)
    else:
        print("加载预训练权重...")
        opt_weights = np.load("optimized_weights.npy")

    # 全库搜索
    print("开始全库搜索...")
    database_imgs = load_database_images(DATABASE_PATH)

    with ProcessPoolExecutor() as executor:
        # 提交任务并建立映射
        future_map = {}
        futures = []
        for idx, db_img in enumerate(database_imgs):
            future = executor.submit(compute_ms_ssim, query_img, db_img, opt_weights)
            futures.append(future)
            future_map[future] = idx

        # 初始化结果
        similarity = [0] * len(database_imgs)

        # 处理完成的任务
        for future in tqdm(as_completed(futures),
                           total=len(futures),
                           desc="图像匹配"):
            idx = future_map[future]
            similarity[idx] = future.result()

    # 结果显示
    matched_idx = np.argmax(similarity)
    matched_angle = generate_angles()[matched_idx]
    print(f"\n{'=' * 40}")
    print(f"最佳匹配航向: {matched_angle}°")
    print(f"相似度得分: {similarity[matched_idx]:.4f}")
    print(f"{'=' * 40}")

    # 显示结果
    matched_img = cv2.imread(os.path.join(DATABASE_PATH, f"{matched_angle:.1f}.png"))

    # plt.figure(figsize=(10, 5))
    # plt.subplot(121), plt.imshow(query_img, cmap='gray'), plt.title("查询图像")
    # plt.subplot(122), plt.imshow(matched_img[:, :, ::-1]), plt.title(f"最佳匹配航向: {matched_angle}°")
    # plt.show()
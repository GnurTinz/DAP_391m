import os
import cv2
import numpy as np

from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
# from sklearn_extra.cluster import KMedoids

def compare_image(x1_path, x2_path):
    img1 = cv2.imread(x1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(x2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        raise AttributeError("Image is none!")
    
    h, w = img1.shape

    img2 = cv2.resize(img2, (w, h))

    score, diff = ssim(img1, img2, data_range=255, full=True)

    # print("Score index:", score)
    
    # cv2.imshow("Different:", diff)
    # cv2.waitKey(0)
    return 1 - score

def k_medoids_pure_numpy(distance_matrix, k, max_iter=100, random_state=42):
    n = distance_matrix.shape[0]

    rng = np.random.default_rng(random_state)
    medoids = rng.choice(n, k, replace=False)
    
    for _ in range(max_iter):
        distances_to_medoids = distance_matrix[:, medoids]
        labels = np.argmin(distances_to_medoids, axis=1)
        
        old_medoids = medoids.copy()
        
        for i in range(k):
            cluster_members = np.where(labels == i)[0]
            if len(cluster_members) == 0:
                continue
            
            sub_distance = distance_matrix[cluster_members[:, None], cluster_members]
            total_distances = np.sum(sub_distance, axis=1)
            
            best_member_idx = np.argmin(total_distances)
            medoids[i] = cluster_members[best_member_idx]
            
        if np.array_equal(np.sort(old_medoids), np.sort(medoids)):
            break
            
    return medoids

def k_medoids_pam_robust(distance_matrix, k, max_iter=100, random_state=42):
    n = distance_matrix.shape[0]
    if k > n:
        raise ValueError("Số lượng K không được lớn hơn tổng số ảnh N!")
        
    # Bước 1: Khởi tạo ngẫu nhiên K ảnh phân biệt làm medoids ban đầu
    rng = np.random.default_rng(random_state)
    medoids = rng.choice(n, k, replace=False)
    
    # Hàm phụ để tính tổng khoảng cách từ tất cả các ảnh đến medoid gần nhất của chúng
    def calculate_total_cost(current_medoids):
        distances_to_medoids = distance_matrix[:, current_medoids]
        min_distances = np.min(distances_to_medoids, axis=1)
        return np.sum(min_distances)
    
    best_cost = calculate_total_cost(medoids)
    
    # Vòng lặp tối ưu hóa tổng thể
    for _ in tqdm(range(max_iter), desc="Optimizing K-Medoids (PAM)"):
        improved = False
        
        # Danh sách các ảnh hiện tại không phải là medoid
        non_medoids = np.setdiff1d(np.arange(n), medoids)
        
        # Thử hoán đổi từng medoid hiện tại với từng ảnh non-medoid
        for m_idx in range(k):
            for non_m in non_medoids:
                # Tạo một tập hợp medoids giả định sau khi hoán đổi
                test_medoids = medoids.copy()
                test_medoids[m_idx] = non_m
                
                # Tính chi phí của tập hợp mới này
                test_cost = calculate_total_cost(test_medoids)
                
                # Nếu việc hoán đổi giúp tổng khoảng cách GIẢM XUỐNG
                if test_cost < best_cost:
                    best_cost = test_cost
                    medoids = test_medoids
                    improved = True
                    break # Thoát ra để cập nhật danh sách non_medoids mới
            if improved:
                break
                
        # Nếu duyệt qua tất cả các cặp hoán đổi mà không làm giảm chi phí nữa thì dừng
        if not improved:
            break
            
    return np.sort(medoids)

if __name__ == "__main__":
    compare_image("E:/palm/data-collection/mother_dataset/person_1/left/11.jpg", "E:/palm/data-collection/mother_dataset/person_1/left/13.jpg")
    # read_full_folder("E:/palm/data-collection/mother_dataset/person_1/left", "E:/palm/data-collection/sub_dataset/script1/person_1/left")
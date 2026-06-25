# Review kỹ thuật ý tưởng PalmPrint Attendance

**Tài liệu review:** Pipeline PalmPrint + file `IDEA.docx`  
**Mục tiêu hệ thống:** điểm danh bằng lòng bàn tay trong bối cảnh **open-set**, có khả năng nhận diện người đã đăng ký, từ chối người chưa đăng ký, giảm false accept, và tăng độ bền trước mẫu nhiễu / mẫu giả / spoof.

---

## 1. Tóm tắt ý tưởng hiện tại

Pipeline hiện tại có thể được hiểu như sau:

1. **Training**
   - Ảnh lòng bàn tay được đưa qua encoder.
   - Encoder xuất ra hai thành phần:
     - `μ ∈ R^d`: vector trung tâm latent / identity representation.
     - `σ ∈ R^d`: độ bất định hoặc độ phân tán trong latent space.
   - Sử dụng reparameterization:
     ```text
     z = μ + σ ⊙ ε,   ε ~ N(0, I)
     ```
   - Decoder tái tạo lại ảnh lòng bàn tay từ `z`.
   - Nhánh Light MLP học phân biệt “similar / not similar” theo tinh thần contrastive learning.

2. **Inference / Sample**
   - Với ảnh đầu vào mới, encoder tạo ra latent distribution.
   - Sinh ra nhiều mẫu gần và xa trong latent space.
   - Tìm một vector hoặc residual `r` bằng tối ưu.
   - Dùng `r` để truy vấn database, tìm ứng viên gần nhất.
   - Verification lần cuối để quyết định accept/reject.

3. **Mục tiêu nghiên cứu**
   - Giải quyết bài toán điểm danh bằng PalmPrint theo hướng open-set.
   - Giảm việc nhận sai người.
   - Sử dụng generative model / VAE / contrastive learning để học latent space tốt hơn.
   - Tăng năng lực reject unknown hoặc mẫu giả.

---

## 2. Nhận xét tổng quan

Ý tưởng có tiềm năng, nhưng nên định vị lại rõ hơn.

Hiện tại pipeline đang trộn ba hướng:

1. **Representation learning**  
   Học embedding lòng bàn tay đủ phân biệt identity.

2. **Generative / probabilistic modeling**  
   Không biểu diễn mỗi ảnh bằng một vector cố định, mà bằng một phân phối:
   ```text
   q(z|x) = N(μ, diag(σ²))
   ```

3. **Open-set verification**  
   Không chỉ phân loại vào người gần nhất, mà còn phải quyết định có đủ tin cậy để accept hay không.

Ba hướng này có thể kết hợp, nhưng nếu không thiết kế chặt sẽ tạo ra pipeline phức tạp, khó chứng minh đóng góp. Nên reformulate project thành:

> **Uncertainty-aware probabilistic PalmPrint verification for open-set attendance.**

Nói cách khác:  
**đóng góp chính không nên là “VAE sinh ảnh lòng bàn tay”, mà là “biểu diễn PalmPrint bằng phân phối có uncertainty, kết hợp contrastive learning và verification hai tầng để tăng khả năng open-set reject”.**

---

## 3. Vấn đề thuật ngữ: ACC cao nhưng EER thấp

Trong idea có nhắc: “Độ chính xác cao nhưng EER thấp” và lý do là “nhận nhiều mẫu sai”.

Chỗ này cần sửa.

- **EER thấp thường là tốt.**
- Nếu hệ thống nhận nhiều mẫu sai, vấn đề thường là:
  - FAR / FMR cao.
  - EER cao.
  - Threshold chọn sai.
  - Closed-set accuracy cao nhưng open-set reject kém.
  - Model luôn ép input vào một identity đã biết, kể cả input unknown.

Vì đây là bài toán điểm danh, metric quan trọng nhất không phải accuracy. Cần đánh giá theo hướng biometric verification / open-set identification.

### Metric nên dùng

| Nhóm metric | Ý nghĩa |
|---|---|
| `FAR / FMR` | Tỷ lệ nhận nhầm người không hợp lệ thành người hợp lệ |
| `FRR / FNMR` | Tỷ lệ từ chối nhầm người hợp lệ |
| `EER` | Điểm cân bằng giữa FAR và FRR |
| `TAR @ FAR=1e-2, 1e-3, 1e-4` | Độ nhận đúng tại mức false accept cố định |
| `AUROC known-vs-unknown` | Khả năng phân biệt known và unknown |
| `FPR95` | False positive rate khi TPR đạt 95% |
| `DIR @ FAR` | Open-set identification: nhận đúng danh tính trong điều kiện kiểm soát false accept |
| `Attack FAR` | Tỷ lệ spoof / synthetic sample được accept |

Với điểm danh, **FAR thấp quan trọng hơn ACC cao**. Một hệ thống accuracy cao nhưng FAR cao vẫn nguy hiểm.

---

## 4. Điểm mạnh của ý tưởng

### 4.1. Nhìn bài toán theo open-set là đúng

Trong điểm danh thực tế, hệ thống không chỉ gặp người đã đăng ký. Nó có thể gặp:

- người chưa đăng ký;
- ảnh lòng bàn tay chất lượng kém;
- ảnh chụp sai vùng ROI;
- ảnh giả;
- ảnh của người khác cố tình giả mạo.

Do đó, closed-set classifier thông thường là không đủ.

### 4.2. Dùng `μ` và `σ` là hướng hợp lý

Biểu diễn mỗi ảnh bằng phân phối Gaussian có lợi thế:

```text
μ: identity signal
σ: uncertainty / ambiguity / sample quality
```

Nếu ảnh rõ, ROI tốt, đường chỉ tay rõ, model nên cho `σ` nhỏ.  
Nếu ảnh mờ, lệch vùng, nhiễu, hoặc gần boundary giữa nhiều identity, model nên cho `σ` lớn.

Điều này phù hợp với bài toán open-set, vì unknown hoặc input kém chất lượng thường có uncertainty cao hơn.

### 4.3. Verification hai tầng là cần thiết

Pipeline có retrieval rồi verification. Đây là thiết kế đúng cho biometric:

1. **Identification / Retrieval**
   - Tìm top-K ứng viên gần nhất trong database.

2. **Verification / Reject**
   - Kiểm tra lại query và candidate.
   - Chỉ accept nếu score vượt threshold và margin đủ rõ.

Thiết kế này tốt hơn việc lấy nearest neighbor rồi accept ngay.

### 4.4. Contrastive learning phù hợp với PalmPrint

PalmPrint có nhiều pattern cục bộ gần giống nhau giữa các người khác nhau. Nếu chỉ dùng classification loss, embedding có thể tốt cho train identity nhưng yếu khi gặp identity mới.

Contrastive / supervised contrastive giúp:

- kéo sample cùng identity lại gần;
- đẩy sample khác identity ra xa;
- tạo latent space có ý nghĩa metric hơn;
- hỗ trợ verification tốt hơn.

---

## 5. Điểm yếu và rủi ro hiện tại

### 5.1. VAE sinh ảnh không đảm bảo học identity

VAE có thể tái tạo ảnh tốt nhưng vẫn học nhầm thông tin không liên quan đến identity:

- illumination;
- background;
- texture camera;
- crop position;
- noise;
- sensor artifact.

Với biometric, reconstruction tốt không đồng nghĩa verification tốt. Nếu đặt `L_rec` quá mạnh, model có thể ưu tiên tái tạo pixel thay vì học đặc trưng nhận dạng.

**Khuyến nghị:** decoder chỉ nên đóng vai trò regularizer, không phải lõi của mô hình.

---

### 5.2. Negative sample bằng Gaussian shift không đảm bảo “khác semantic”

Trong pipeline, phần sample “not similar” có thể được hiểu là lấy noise từ phân phối bị dịch:

```text
ε ~ N(δ, I)
```

Cách này không đảm bảo sample sinh ra là khác identity. Trong biometric latent space, semantic identity không nhất thiết tuyến tính theo hướng dịch Gaussian.

Negative nên được tạo từ dữ liệu thật hoặc prototype thật:

- identity khác trong batch;
- hard negative gần nhất nhưng khác label;
- memory bank negatives;
- prototype của người khác;
- interpolation giữa hai identity khác nhau.

---

### 5.3. Test-time optimization của `r` có nguy cơ overfit

Phần “finding r” là ý tưởng thú vị nhưng rủi ro cao.

Nếu tại inference cho phép tối ưu `r` quá tự do, mô hình có thể tìm ra một `r` khiến verifier tin rằng query giống một người trong database, kể cả query là unknown. Điều này có thể làm tăng false accept.

Đặc biệt với open-set, test-time optimization phải có ràng buộc rất mạnh. Nếu không, nó có thể biến thành cơ chế “ép query về identity gần nhất”.

---

### 5.4. Pipeline nhiều module, khó chứng minh đóng góp

Hiện có quá nhiều thành phần:

- encoder;
- `μ`;
- `σ`;
- sampling;
- decoder;
- contrastive MLP;
- generated similar/dissimilar samples;
- optimizing `r`;
- attention / database retrieval;
- verification MLP.

Nếu đưa tất cả cùng lúc, khi kết quả tốt hơn baseline sẽ khó trả lời:

> Thành phần nào thật sự giúp cải thiện?

Cần thiết kế ablation nghiêm túc.

---

### 5.5. Chưa tách rõ recognition và anti-spoofing

Idea có nhắc hệ thống dễ bị tấn công bằng lòng bàn tay giả. Tuy nhiên, generative embedding không tự động giải quyết presentation attack.

Cần tách hai bài toán:

1. **PalmPrint recognition / verification**
   - Người này là ai?
   - Có phải người đã đăng ký không?

2. **Presentation attack detection / liveness**
   - Đây là lòng bàn tay thật hay ảnh giả / bản in / màn hình / synthetic image?

Nếu muốn claim chống spoof, cần có module PAD hoặc thí nghiệm attack riêng.

---

## 6. Kiến trúc nên chọn

### 6.1. Không nên bắt đầu bằng diffusion model

Diffusion model mạnh cho generation, nhưng không phải lựa chọn đầu tiên cho bài toán này, vì:

- nặng;
- khó train;
- khó deploy real-time cho điểm danh;
- khó chứng minh nó cải thiện verification;
- dễ khiến project lệch sang bài toán sinh ảnh.

Diffusion chỉ nên dùng sau nếu nhóm muốn:

- sinh dữ liệu palmprint synthetic;
- tạo spoof attack set;
- augmentation nâng cao;
- nghiên cứu presentation attack.

---

### 6.2. Kiến trúc chính nên là probabilistic embedding

Đề xuất core model:

```text
x -> Encoder -> μ, logσ
z = μ + σ ⊙ ε
```

Trong đó:

- `μ` là embedding nhận dạng chính.
- `σ` là uncertainty.
- `z` dùng để sampling / regularization / Monte Carlo scoring.
- decoder là nhánh phụ, có thể bật/tắt trong ablation.

### 6.3. Backbone encoder

Có thể thử theo thứ tự:

1. **ResNet / EfficientNet**
   - Baseline mạnh, dễ train.
   - Phù hợp nếu dataset không quá lớn.

2. **ConvNeXt / Swin Transformer**
   - Tốt hơn với texture pattern nếu dữ liệu đủ.

3. **ViT nhỏ**
   - Cần nhiều data/augmentation hơn.
   - Có thể mạnh nếu ROI chuẩn và dataset lớn.

Khuyến nghị thực tế:  
**bắt đầu với ResNet-18/34 hoặc EfficientNet-B0/B1**, sau đó mới thử backbone mạnh hơn.

---

## 7. Cách sample giống và khác

### 7.1. Similar sample

Similar sample nên đến từ các nguồn có căn cứ semantic:

#### Nguồn 1: cùng ảnh, khác augmentation

```text
x_i^1 = aug1(x_i)
x_i^2 = aug2(x_i)
```

Hai view này là positive pair.

#### Nguồn 2: cùng identity

Nếu có nhiều ảnh cho một người:

```text
x_i, x_j, y_i = y_j
```

Đây là positive mạnh hơn augmentation.

#### Nguồn 3: sample từ Gaussian latent

```text
z_i^+ = μ_i + σ_i ⊙ ε,   ε ~ N(0, I)
```

Sampling này giúp ước lượng vùng latent hợp lệ quanh input.

---

### 7.2. Dissimilar sample

Negative nên ưu tiên hard negative thay vì random noise.

#### Loại 1: batch negative

```text
y_i ≠ y_j
```

Dễ triển khai, dùng trong contrastive loss.

#### Loại 2: hard negative

Chọn mẫu khác identity nhưng gần query trong latent space:

```text
j = argmin distance(μ_i, μ_j), y_i ≠ y_j
```

Đây là loại negative quan trọng nhất cho bài toán “khoảng cách gần nhưng semantic khác”.

#### Loại 3: memory bank negative

Lưu embedding/prototype từ nhiều batch trước, chọn negative gần nhất.

#### Loại 4: prototype negative

So với prototype của identity khác:

```text
P_c = mean(μ_c)
```

Negative là các prototype gần nhưng khác identity.

#### Loại 5: boundary interpolation

Tạo sample ở vùng biên giữa hai identity khác nhau:

```text
z^- = α μ_i + (1 - α) μ_j,   y_i ≠ y_j
```

Cách này có thể dùng để dạy verifier reject vùng không chắc chắn.

---

### 7.3. Không nên định nghĩa negative chỉ bằng `ε ~ N(δ, I)`

Lý do:

- dịch noise không tương ứng với đổi identity;
- có thể sinh sample vô nghĩa;
- có thể làm model học artifact thay vì semantic;
- khó giải thích trong paper/report.

Có thể giữ Gaussian-shift negative như một ablation, nhưng không nên là cơ chế chính.

---

## 8. Nên dùng `μ`, `σ`, hay kết hợp cả hai?

### 8.1. Vai trò đề xuất

| Thành phần | Vai trò |
|---|---|
| `μ` | embedding nhận dạng chính |
| `σ` | uncertainty / chất lượng / độ mơ hồ |
| `z` | sample latent dùng để regularize và scoring |
| `r` | residual / representation refined bởi database hoặc verifier |

### 8.2. Contrastive loss nên đặt chủ yếu trên `μ`

Công thức:

```text
p_i = Projection(μ_i)
L_supcon = SupCon(p_i, y_i)
```

Hoặc:

```text
p_i = Projection(z_i)
```

Nhưng nếu dùng `z`, cần đảm bảo `σ` không bị collapse hoặc phình quá lớn.

### 8.3. Pair verification nên dùng cả `μ` và `σ`

Với query `i` và candidate `j`, tạo feature:

```text
φ_ij = [
  μ_i,
  μ_j,
  |μ_i - μ_j|,
  μ_i ⊙ μ_j,
  σ_i,
  σ_j,
  |σ_i - σ_j|
]
```

Sau đó:

```text
score_ij = MLP(φ_ij)
```

Train bằng binary cross entropy:

```text
L_pair = BCE(score_ij, 1[y_i = y_j])
```

Cách này tốt hơn việc chỉ tính cosine similarity, vì verifier có thể học tương tác giữa identity distance và uncertainty.

---

## 9. Loss function đề xuất

Tổng loss:

```text
L_total =
  λ_id   L_id
+ λ_con  L_supcon
+ λ_pair L_pair
+ λ_rec  L_rec
+ β      L_KL
+ λ_unc  L_unc
```

### 9.1. `L_id`

Dùng classification/proxy loss để tạo embedding phân biệt identity.

Có thể dùng:

- Cross-Entropy baseline;
- ArcFace;
- CosFace;
- ProxyAnchor.

Nếu có đủ label identity, nên dùng ArcFace hoặc CosFace thay cho softmax thường.

### 9.2. `L_supcon`

Dùng supervised contrastive learning:

- positive: cùng identity / augmentation cùng ảnh;
- negative: khác identity;
- ưu tiên hard negatives.

### 9.3. `L_pair`

Dùng để train trực tiếp module verification:

```text
same/different pair -> MLP -> probability same identity
```

Loss này rất quan trọng vì bài toán cuối là verification.

### 9.4. `L_rec`

Reconstruction loss từ decoder:

```text
L_rec = L1(x, x_hat)
```

Có thể kết hợp thêm SSIM nếu ROI ảnh ổn định.

Không nên đặt `λ_rec` quá lớn. Reconstruction chỉ là regularizer.

### 9.5. `L_KL`

VAE-style KL:

```text
L_KL = KL(q(z|x) || N(0, I))
```

Cần cẩn thận với `β`. Nếu `β` quá cao, latent bị ép về chuẩn Gaussian và mất identity information.

### 9.6. `L_unc`

Dùng để kiểm soát `σ`:

- tránh `σ -> 0` toàn bộ;
- tránh `σ` phình quá lớn;
- khuyến khích mẫu khó có uncertainty cao hơn mẫu dễ.

Một hướng đơn giản:

```text
L_unc = penalty(mean(logσ), lower_bound, upper_bound)
```

Hoặc học uncertainty theo difficulty:

```text
hard / misclassified samples -> σ lớn hơn
easy / clean samples -> σ nhỏ hơn
```

---

## 10. Chiến lược training đề xuất

Không nên train tất cả từ đầu cùng lúc. Nên dùng staged training.

### Stage 1: Train deterministic baseline

```text
x -> Encoder -> embedding
```

Loss:

```text
L = L_id + L_supcon
```

Mục tiêu:

- có baseline mạnh;
- kiểm tra ROI/data;
- đo closed-set và open-set cơ bản.

### Stage 2: Thêm probabilistic head

```text
x -> Encoder -> μ, logσ
```

Loss:

```text
L = L_id(μ) + L_supcon(μ) + L_unc
```

Mục tiêu:

- kiểm tra `σ` có ý nghĩa không;
- đo uncertainty trên clean/noisy/unknown samples.

### Stage 3: Thêm pair verifier

```text
(μ_i, σ_i), (μ_j, σ_j) -> MLP -> same/different
```

Loss:

```text
L = L_id + L_supcon + L_pair + L_unc
```

Mục tiêu:

- cải thiện verification;
- calibrate threshold.

### Stage 4: Thêm decoder VAE như regularizer

```text
z -> Decoder -> x_hat
```

Loss:

```text
L = previous_losses + λ_rec L_rec + β L_KL
```

Mục tiêu:

- kiểm tra decoder có giúp open-set hay không;
- nếu không giúp, bỏ decoder.

### Stage 5: Thử cơ chế `r`

Chỉ thử sau khi baseline probabilistic verification đã ổn.  
Không nên đưa `r` vào từ đầu.

---

## 11. Cơ chế `r`: nên sửa như thế nào?

### 11.1. Không nên tối ưu `r` tự do

Công thức hiện tại có thể hiểu là:

```text
r* = argmin_r L(MLP(X_new + r), label)
```

Rủi ro:

- overfit query;
- kéo unknown về known identity;
- tăng FAR;
- khó deploy real-time;
- khó giải thích.

### 11.2. Phương án khuyến nghị A: bỏ `r`, dùng Monte Carlo scoring

Với query:

```text
q(z|x_q) = N(μ_q, σ_q²)
```

Sample:

```text
z_q^k ~ q(z|x_q), k = 1..K
```

Score với candidate `c`:

```text
Score(q, c) = mean_k Verifier(z_q^k, P_c)
```

Ưu điểm:

- đơn giản;
- ít rủi ro;
- phù hợp uncertainty;
- dễ ablation;
- không có test-time optimization.

Đây nên là baseline chính.

---

### 11.3. Phương án khuyến nghị B: `r` là attention từ top-K database

Thay vì tối ưu `r` tự do:

```text
topK = retrieve(μ_q, database)
r = Attention(μ_q, P_topK)
```

Sau đó verify:

```text
score = MLP([μ_q, r, |μ_q-r|, μ_q⊙r, σ_q])
```

Ưu điểm:

- `r` có nguồn gốc từ database;
- không trôi khỏi query;
- dễ giải thích hơn;
- phù hợp với retrieval + verification.

---

### 11.4. Phương án C: giữ optimization nhưng ràng buộc mạnh

Nếu vẫn muốn giữ “finding r”, cần công thức rõ:

```text
r_0 = μ_q

r* = argmin_r [
  L_verify(r, P_topK)
  + α ||r - μ_q||²
  + γ U(r)
]
```

Điều kiện dừng:

- số bước tối đa: 5–20;
- `||r_t - r_{t-1}|| < ε`;
- top-1 candidate không đổi trong `m` bước;
- score vượt threshold;
- margin top1-top2 đủ lớn;
- nếu uncertainty cao thì không optimize tiếp mà reject;
- ràng buộc `||r - μ_q|| < δ`.

Tuy nhiên, đây nên là nhánh research phụ, không phải core pipeline đầu tiên.

---

## 12. Verification nên thiết kế thế nào?

### 12.1. Database enrollment

Với mỗi identity `c`, lưu nhiều mẫu:

```text
D_c = { (μ_c1, σ_c1), ..., (μ_cm, σ_cm) }
```

Hoặc prototype:

```text
μ̄_c = mean_k μ_ck
σ̄_c = aggregate_k σ_ck
```

Nếu có nhiều ảnh enrollment mỗi người, nên giữ cả set thay vì chỉ mean, vì palmprint có thể thay đổi theo pose, ánh sáng, áp lực bàn tay.

---

### 12.2. Retrieval score

Một score đơn giản:

```text
D(q, c) = cosine_distance(μ_q, μ̄_c) + η mean(σ_q)
```

Hoặc:

```text
D(q, c) = ||μ_q - μ̄_c||² + η ||σ_q - σ̄_c||²
```

Có thể dùng top-K candidate.

---

### 12.3. Verification score

Với top-K candidate:

```text
score_c = Verifier(q, c)
```

Accept nếu thỏa tất cả điều kiện:

```text
score_top1 > τ
score_top1 - score_top2 > margin
uncertainty(q) < u_max
quality(q) > q_min
```

Nếu không:

```text
reject / unknown / retry capture
```

Đây là cơ chế reject cần thiết cho open-set.

---

## 13. Nên concat hay cộng trực tiếp?

Khuyến nghị: **concat trước, không cộng trực tiếp.**

### 13.1. Vì sao không nên cộng trực tiếp?

Cộng:

```text
h = x + r
```

chỉ hợp lý nếu:

- `x` và `r` cùng không gian;
- từng chiều có cùng ý nghĩa;
- scale tương thích;
- `r` thực sự là residual trong cùng manifold.

Hiện tại các điều kiện này chưa được chứng minh.

### 13.2. Cách concat khuyến nghị

Dùng feature pair:

```text
h = [
  μ_q,
  r,
  |μ_q - r|,
  μ_q ⊙ r,
  σ_q
]
```

hoặc nếu so với candidate:

```text
h = [
  μ_q,
  μ_c,
  |μ_q - μ_c|,
  μ_q ⊙ μ_c,
  σ_q,
  σ_c,
  |σ_q - σ_c|
]
```

Sau đó đưa qua MLP.

Ưu điểm:

- không làm mất thông tin;
- MLP tự học tương tác;
- dễ ablation;
- dễ giải thích.

Sau khi concat tốt, có thể thử gated fusion:

```text
h = μ_q + gate(r) ⊙ value(r)
```

Nhưng gated fusion nên là bước sau.

---

## 14. Thiết kế thí nghiệm

### 14.1. Dataset split

Cần split theo identity, không split ngẫu nhiên theo ảnh.

Ví dụ:

```text
Train identities:      60%
Validation identities: 20%
Test identities:       20%
```

Trong test:

- Known-test: identity có enrollment trong database.
- Unknown-test: identity không có trong database.

Đây là điều kiện bắt buộc cho open-set.

---

### 14.2. Experiment 1: Closed-set baseline

Mục tiêu: chứng minh backbone và embedding học được identity.

So sánh:

| Model | Mục đích |
|---|---|
| CNN + Softmax | baseline thấp |
| CNN + ArcFace/CosFace | discriminative baseline |
| CNN + Triplet | metric baseline |
| CNN + SupCon | contrastive baseline |
| Proposed probabilistic encoder | mô hình đề xuất |

Metric:

- Rank-1 accuracy;
- Rank-5 accuracy;
- intra/inter-class distance;
- embedding visualization.

---

### 14.3. Experiment 2: Open-set verification

Mục tiêu: chứng minh hệ thống reject tốt unknown.

Protocol:

1. Enrollment bằng một hoặc nhiều mẫu mỗi identity known.
2. Query gồm:
   - genuine known;
   - impostor known;
   - unknown identities.
3. Tính score và threshold.

Metric:

- EER;
- FAR/FMR;
- FRR/FNMR;
- TAR @ FAR=1e-2;
- TAR @ FAR=1e-3;
- TAR @ FAR=1e-4;
- AUROC known-vs-unknown;
- FPR95;
- DIR @ FAR.

---

### 14.4. Experiment 3: Ablation probabilistic component

| Variant | μ | σ | Sampling | Decoder | Pair verifier | Mục đích |
|---|---:|---:|---:|---:|---:|---|
| Deterministic | yes | no | no | no | optional | baseline |
| μ + σ | yes | yes | no | no | yes | kiểm tra uncertainty |
| μ + σ + sampling | yes | yes | yes | no | yes | kiểm tra Monte Carlo |
| VAE only | yes | yes | yes | yes | no | kiểm tra reconstruction |
| Full | yes | yes | yes | yes | yes | mô hình đầy đủ |

Kỳ vọng:

- `μ + σ` phải cải thiện open-set hơn deterministic.
- Sampling phải cải thiện calibration hoặc TAR@FAR thấp.
- Decoder chỉ nên giữ nếu giúp metric open-set.

---

### 14.5. Experiment 4: Ablation negative sampling

So sánh:

| Negative type | Mô tả |
|---|---|
| random batch negative | khác identity trong batch |
| hard negative | khác identity nhưng gần nhất |
| memory bank negative | negative từ queue/prototype |
| Gaussian-shift negative | dịch noise Gaussian |
| boundary interpolation | nội suy giữa hai identity |

Mục tiêu: chứng minh loại negative nào thật sự giúp giảm false accept.

---

### 14.6. Experiment 5: Ablation `r`

So sánh:

| Variant | Mô tả |
|---|---|
| No-r | chỉ dùng μ, σ |
| MC scoring | sample z nhiều lần, lấy average score |
| Attention-r | `r` từ top-K database |
| Optimized-r | tối ưu `r` tại inference |
| Optimized-r + constraints | tối ưu có regularization |

Metric quan trọng nhất:

- FAR;
- TAR @ FAR=1e-3;
- unknown AUROC;
- latency;
- số trường hợp unknown bị kéo nhầm về known.

Nếu `optimized-r` tăng accuracy nhưng tăng FAR, không nên dùng cho điểm danh.

---

### 14.7. Experiment 6: Robustness và attack

Test các điều kiện:

- blur;
- illumination shift;
- crop lệch;
- low resolution;
- sensor noise;
- print attack;
- screen replay;
- synthetic palmprint nếu có.

Metric:

- FAR under attack;
- reject rate;
- score distribution của genuine / impostor / unknown / attack;
- nếu có PAD: APCER, BPCER, ACER.

---

## 15. Roadmap triển khai đề xuất

### Phase 0: Chuẩn hóa data

- ROI extraction ổn định.
- Normalize ảnh.
- Kiểm tra số mẫu mỗi identity.
- Split theo identity.
- Tạo enrollment/query protocol.

### Phase 1: Baseline mạnh

Train:

```text
Encoder + ArcFace/CosFace/SupCon
```

Output:

- closed-set metric;
- verification metric;
- threshold baseline.

### Phase 2: Probabilistic embedding

Train:

```text
Encoder -> μ, logσ
```

Output:

- uncertainty analysis;
- open-set verification;
- compare deterministic vs probabilistic.

### Phase 3: Pair verifier

Train:

```text
Verifier([μ_i, μ_j, |μ_i-μ_j|, μ_i⊙μ_j, σ_i, σ_j])
```

Output:

- score calibration;
- FAR reduction;
- TAR@FAR improvement.

### Phase 4: Sampling / Monte Carlo scoring

Test:

```text
K = 5, 10, 20, 50
```

Output:

- trade-off accuracy vs latency;
- stability score;
- calibration improvement.

### Phase 5: Decoder VAE

Add:

```text
z -> Decoder -> x_hat
```

Output:

- ablation whether reconstruction helps;
- nếu không giúp thì bỏ.

### Phase 6: `r` mechanism

Test only after previous phases are stable.

Priority:

1. no-r;
2. MC scoring;
3. attention-r;
4. optimized-r with constraints.

---

## 16. Phiên bản pipeline đề xuất

### 16.1. Training pipeline

```text
Input palm ROI
    ↓
Encoder backbone
    ↓
μ, logσ
    ↓
z = μ + σ ⊙ ε
    ↓
[Optional] Decoder reconstruction
```

Parallel heads:

```text
μ -> Projection head -> SupCon loss
μ -> ID classifier / ArcFace
(μ_i, σ_i, μ_j, σ_j) -> Pair verifier -> BCE loss
```

Total loss:

```text
L_total =
  λ_id   L_id
+ λ_con  L_supcon
+ λ_pair L_pair
+ λ_rec  L_rec
+ β      L_KL
+ λ_unc  L_unc
```

---

### 16.2. Enrollment

For each registered person:

```text
store {
  identity_id,
  μ samples,
  σ samples,
  prototype μ̄,
  prototype σ̄,
  quality statistics
}
```

---

### 16.3. Inference

```text
Input x_q
    ↓
ROI + preprocessing
    ↓
Encoder -> μ_q, σ_q
    ↓
Quality / uncertainty check
    ↓
Retrieve top-K candidates
    ↓
Pair verifier / MC scoring
    ↓
Accept if:
  score_top1 > τ
  margin_top1_top2 > m
  uncertainty < u_max
Else:
  reject / retry
```

---

## 17. Các câu hỏi trong IDEA và câu trả lời đề xuất

### Q1. Generative model nên chọn model nào?

**Chọn probabilistic embedding / VAE-inspired encoder trước.**  
Không nên bắt đầu bằng diffusion. Decoder VAE chỉ nên là regularizer.

---

### Q2. Cách sample giống và khác?

Similar:

- augmentation cùng ảnh;
- ảnh cùng identity;
- sample từ `N(μ, σ²)`.

Dissimilar:

- khác identity trong batch;
- hard negative;
- memory bank negative;
- prototype negative;
- boundary interpolation.

Không nên dùng Gaussian-shift negative làm cơ chế chính.

---

### Q3. Sample theo `μ`, `σ`, hay kết hợp?

Kết hợp:

- `μ` dùng cho identity;
- `σ` dùng cho uncertainty;
- `z = μ + σ⊙ε` dùng cho sampling.

Contrastive nên đặt chủ yếu trên `μ` hoặc projection của `μ`. Pair verifier nên dùng cả `μ` và `σ`.

---

### Q4. Train generative model thế nào?

Dùng tổng loss:

```text
L_total =
  λ_id L_id
+ λ_con L_supcon
+ λ_pair L_pair
+ λ_rec L_rec
+ β KL
+ λ_unc L_unc
```

Nên train theo stage. Không train full model từ đầu.

---

### Q5. Khi nào dừng tìm `r`?

Nếu vẫn tối ưu `r`, dừng khi:

- đạt số bước tối đa;
- `||r_t-r_{t-1}|| < ε`;
- top-1 candidate ổn định;
- score vượt threshold;
- margin đủ lớn;
- uncertainty quá cao thì reject thay vì tiếp tục optimize;
- `||r-μ_q||` vượt giới hạn thì stop/reject.

Tuy nhiên, khuyến nghị ban đầu: **không dùng optimized-r**, dùng MC scoring hoặc attention-r.

---

### Q6. Verification như thế nào?

Dùng hai tầng:

1. Retrieve top-K bằng `μ`, có penalty uncertainty.
2. Pair verifier dùng `[μ_q, μ_c, |μ_q-μ_c|, μ_q⊙μ_c, σ_q, σ_c]`.

Accept nếu:

```text
score > τ
margin > m
uncertainty < u_max
quality > q_min
```

---

### Q7. Nên concat với đầu vào hay cộng trực tiếp?

**Concat.**

Công thức đề xuất:

```text
h = [μ_q, r, |μ_q-r|, μ_q⊙r, σ_q]
```

hoặc với candidate:

```text
h = [μ_q, μ_c, |μ_q-μ_c|, μ_q⊙μ_c, σ_q, σ_c, |σ_q-σ_c|]
```

Không nên cộng trực tiếp trừ khi chứng minh được `r` là residual cùng manifold với `μ`.

---

## 18. Định vị research contribution

Nên viết contribution theo hướng:

1. **Probabilistic PalmPrint embedding**
   - Biểu diễn mỗi palmprint bằng Gaussian latent distribution thay vì deterministic vector.

2. **Uncertainty-aware open-set verification**
   - Dùng uncertainty để reject mẫu mơ hồ / unknown / chất lượng thấp.

3. **Contrastive learning with hard negatives**
   - Giải quyết vấn đề “gần trong embedding nhưng khác semantic”.

4. **Two-stage attendance decision**
   - Retrieval top-K + verification + threshold + margin.

5. **Generative regularization**
   - Decoder/VAE giúp regularize latent, nhưng không phải lõi bắt buộc.

Không nên claim quá mạnh rằng model chống spoof nếu chưa có thí nghiệm PAD rõ ràng.

---

## 19. Checklist trước khi viết paper/report

### Data

- [ ] ROI extraction ổn định.
- [ ] Split theo identity.
- [ ] Có known/unknown test protocol.
- [ ] Có enrollment/query split.
- [ ] Có augmentations hợp lý.

### Model

- [ ] Baseline deterministic.
- [ ] Probabilistic encoder.
- [ ] SupCon / ArcFace.
- [ ] Pair verifier.
- [ ] Optional decoder.
- [ ] Optional `r`.

### Evaluation

- [ ] Closed-set Rank-1/Rank-5.
- [ ] Verification EER.
- [ ] TAR@FAR.
- [ ] Open-set AUROC.
- [ ] FPR95.
- [ ] DIR@FAR.
- [ ] Latency.
- [ ] Ablation từng module.

### Risk

- [ ] Kiểm tra FAR trên unknown.
- [ ] Kiểm tra false accept với hard negatives.
- [ ] Kiểm tra input chất lượng thấp.
- [ ] Kiểm tra attack nếu claim anti-spoof.
- [ ] Calibration threshold trên validation, không tune trên test.

---

## 20. Kết luận

Ý tưởng có hướng nghiên cứu tốt, đặc biệt ở việc đưa PalmPrint attendance về open-set verification và dùng latent distribution `N(μ, σ²)` để mô hình hóa uncertainty.

Tuy nhiên, nên chỉnh trọng tâm:

```text
Không phải:
"VAE sinh ảnh lòng bàn tay để nhận diện"

Mà là:
"Probabilistic contrastive PalmPrint embedding with uncertainty-aware open-set verification"
```

Pipeline nên được đơn giản hóa theo thứ tự:

1. deterministic baseline;
2. probabilistic embedding;
3. pair verifier;
4. Monte Carlo scoring;
5. optional VAE decoder;
6. optional `r`.

Thành phần rủi ro nhất hiện tại là **test-time optimization của `r`**. Nên thay bằng Monte Carlo scoring hoặc attention từ top-K database trước. Nếu vẫn giữ `r`, cần regularization và điều kiện dừng chặt để tránh tăng false accept.

Hướng triển khai tối thiểu nên là:

```text
Encoder -> μ, σ
SupCon/ArcFace training
Pair verifier using μ and σ
Open-set threshold + margin + uncertainty reject
```

Sau khi baseline này có kết quả rõ, nhóm mới nên thêm decoder VAE và cơ chế `r` để chứng minh đóng góp bổ sung.

"""Paper Report page – full overview of the research paper content and findings."""

import os
import streamlit as st
from PIL import Image

IMPLEMENT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "implement-idea")
ARCH_IMG    = os.path.join(IMPLEMENT_DIR, "report", "Palm_Generative_Approach.drawio.png")
OVERALL_IMG = os.path.join(IMPLEMENT_DIR, "report", "overall.png")
DETAIL_IMG  = os.path.join(IMPLEMENT_DIR, "report", "detail-block.png")
GRADIENT_IMG = os.path.join(IMPLEMENT_DIR, "gradient_flow.png")
PIPELINE_IMG = os.path.join(IMPLEMENT_DIR, "pipeline-general.jpg")


def _img(path, caption="", width=None):
    if os.path.exists(path):
        img = Image.open(path)
        if width:
            ratio = width / img.width
            img = img.resize((width, int(img.height * ratio)))
        st.image(img, caption=caption, use_container_width=(width is None))
    else:
        st.caption(f"_(Image not found: {os.path.basename(path)})_")


def render():
    st.markdown("""
    <div class="page-hero">
        <div class="page-title">📋 Research Paper Report</div>
        <div class="page-subtitle">
            Full overview of the paper <b>«Probabilistic Palmprint Embedding with
            Generative Regularization for Open-Set Identification»</b>
            aligned with the documents in <code>implement-idea/</code>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Navigation tabs ───────────────────────────────────────────────────────
    tabs = st.tabs([
        "📌 Problem & Motivation",
        "🏗️ Architecture",
        "⚖️ Loss Functions",
        "📅 Training Strategy",
        "⚡ Inference Modes",
        "🔬 Ablation Design",
        "📊 Key Results",
        "🚦 Open-Set Rejection",
    ])

    # ═════════════════════════════════════════════════════════════════════════
    # Tab 0 – Problem & Motivation
    # ═════════════════════════════════════════════════════════════════════════
    with tabs[0]:
        st.markdown('<div class="section-header">🎯 Bài Toán</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        Nhận dạng lòng bàn tay (Palmprint) trong bối cảnh <b>Open-Set</b>:
        hệ thống phải <b>nhận ra đúng người đã đăng ký</b> và đồng thời
        <b>từ chối người chưa đăng ký (stranger)</b> hay ảnh chất lượng kém.
        Closed-set classifier truyền thống thất bại trong bối cảnh này vì luôn ép
        input về một trong các lớp đã biết, kể cả khi input là unknown.
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            **Thách thức thực tế:**
            - 🌫️ Ảnh mờ, nhiễu, lệch vùng ROI
            - 🙈 Người chưa đăng ký (stranger) xuất hiện
            - 🎭 Tấn công giả mạo (spoof/impostor)
            - 📷 Domain shift giữa các phiên thu thập (session drift)
            - ✋ Tay trái vs. tay phải (cross-hand generalization)
            """)
        with c2:
            st.markdown("""
            **Tại sao cần Probabilistic Embedding?**
            - Vector cố định (deterministic) không biểu diễn được sự không chắc chắn
            - σ² cao → ảnh nhiễu / mờ / vùng lạ → nên reject
            - σ² thấp → ảnh sạch, rõ ràng → tin tưởng cao hơn
            - Phân phối N(μ, σ²) cho phép so sánh bằng KL divergence
            """)

        st.markdown('<div class="section-header">📏 Metrics Quan Trọng</div>', unsafe_allow_html=True)
        st.markdown("""
        | Metric | Ý nghĩa | Hướng tốt |
        |--------|---------|-----------|
        | **Rank-1 Accuracy** (Closed & Open) | % probe được match đúng top-1 | ↑ cao hơn tốt |
        | **EER** (Equal Error Rate) | Điểm cân bằng FAR = FRR | ↓ thấp hơn tốt |
        | **AUROC** (Known vs. Unknown) | Khả năng phân biệt enrolled và stranger | ↑ cao hơn tốt |
        | **DIR @ FAR=1%** | Detection & Identification Rate tại mức false accept 1% | ↑ cao hơn tốt |
        | **OSCR** | Open-Set Classification Rate — kết hợp identification và rejection | ↑ cao hơn tốt |
        | **Open KL-Gate EER** | EER khi dùng KL divergence làm rejection gate | ↓ thấp hơn tốt |
        | **Open Uncertainty EER** | EER khi dùng σ² làm rejection gate | ↓ thấp hơn tốt |

        > **FAR thấp quan trọng hơn Accuracy cao** trong bài toán điểm danh thực tế.
        > Một hệ thống accuracy cao nhưng FAR cao vẫn nguy hiểm.
        """)

    # ═════════════════════════════════════════════════════════════════════════
    # Tab 1 – Architecture
    # ═════════════════════════════════════════════════════════════════════════
    with tabs[1]:
        st.markdown('<div class="section-header">🏗️ Kiến Trúc Tổng Thể</div>', unsafe_allow_html=True)

        ac1, ac2 = st.columns([1, 1])
        with ac1:
            _img(ARCH_IMG, "Probabilistic U-Net + PalmEncoder + MLP Projector")
        with ac2:
            st.markdown("""
            **Luồng Training:**

            ```
            x (Palm ROI 128×128)
                ↓
            PalmEncoder (CCNet / ResNet18 / PalmNet)
                ↓
            μ ∈ ℝᵈ  +  log σ² ∈ ℝᵈ
                ↓
            z = μ + σ ⊙ ε  (ε ~ N(0,I))  ← reparameterization
                ↓
            ┌─────────────────┬───────────────────────────┐
            │  MLP Projector  │  U-Net Decoder (FiLM on z) │
            │       ↓         │       ↓                   │
            │   ArcFace Loss  │  Reconstruction x̂         │
            └─────────────────┴───────────────────────────┘
            ```

            **Luồng Inference (decoder removed):**
            ```
            x → PalmEncoder → μ, σ → Matching / Rejection
            ```
            """)

        st.markdown('<div class="section-header">🔩 FiLM Conditioning trong U-Net Decoder</div>', unsafe_allow_html=True)
        fc1, fc2 = st.columns([1, 1])
        with fc1:
            _img(DETAIL_IMG, "FiLM (Feature-wise Linear Modulation) + UpBlock")
        with fc2:
            st.markdown("""
            **FiLM** (Feature-wise Linear Modulation) cho phép vector z
            điều chỉnh từng feature map trong decoder:

            ```
            FiLM(h, z) = γ(z) ⊙ h + β(z)
            ```

            - γ, β được sinh ra từ z qua một MLP nhỏ
            - Mỗi UpBlock trong U-Net được conditioned riêng biệt
            - Điều này giúp decoder tái tạo đúng cấu trúc vân tay
              ứng với mẫu cụ thể được encode, không phải mean
            """)

        st.markdown('<div class="section-header">🔧 Backbone Encoders</div>', unsafe_allow_html=True)
        bc = st.columns(3)
        backbones = [
            ("ResNet18", "🏆 Best overall accuracy",
             """- ImageNet pretrained weights
- Multi-scale spatial feature extraction
- Strong blur/noise robustness
- Heavy params (decoder dropped at inference)
- **unet_resnet** in config"""),
            ("CCNet (Criss-Cross)", "🔬 Experimental / global context",
             """- Criss-Cross Attention → global palm structure
- Captures long-range dependencies
- High computational cost (attention matrix)
- Palmprint is local-texture heavy → limited gain
- **unet_ccnet** in config"""),
            ("PalmNet / PalmNet-Gabor", "🚀 Lightweight / edge deployment",
             """- Gabor-initialized Conv filters (texture prior)
- Ultra-lightweight (few thousand params)
- Real-time inference on embedded devices
- Sensitive to ROI alignment quality
- **unet_palmnet** / **unet_palmnet_gabor**"""),
        ]
        for col, (name, status, bullets) in zip(bc, backbones):
            col.markdown(f"""
            **{name}**
            _{status}_
            {bullets}
            """)

    # ═════════════════════════════════════════════════════════════════════════
    # Tab 2 – Loss Functions
    # ═════════════════════════════════════════════════════════════════════════
    with tabs[2]:
        st.markdown('<div class="section-header">⚖️ Hàm Mục Tiêu Kết Hợp</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box" style="font-size:1rem; font-family:'JetBrains Mono',monospace; line-height:2.2;">
        <b>L<sub>total</sub></b> = λ<sub>arc</sub>·<b>L<sub>ArcFace</sub></b>
                   + λ<sub>rec</sub>·<b>L<sub>recon</sub></b>
                   + β·<b>L<sub>KL</sub></b>
                   + λ<sub>unc</sub>·<b>L<sub>unc</sub></b>
        </div>
        """, unsafe_allow_html=True)

        for name, badge, formula, role, detail in [
            ("L_ArcFace", "badge-violet",
             "L_arc = −log[exp(s·cos(θ_yi+m)) / (exp(s·cos(θ_yi+m)) + Σⱼ≠yi exp(s·cos(θⱼ)))]",
             "Identity discrimination trên projected space",
             """ArcFace áp dụng **angular margin m** để ép các identity khác nhau ra xa trên hypersphere.
             Được tính trên **proj(μ)** (MLP projection của latent mean).
             Đây là loss chính cho việc học đặc trưng nhận dạng (discriminative objective).
             Thiết kế tách biệt latent_dim và proj_dim giải quyết gradient conflict."""),
            ("L_recon", "badge-cyan",
             "L_rec = L1(x, Decoder(z))  hoặc  MSE(x, x̂)",
             "Generative Regularization qua tái tạo ảnh",
             """Decoder (U-Net FiLM-conditioned) cố gắng tái tạo ảnh gốc từ z.
             Điều này **ép latent code phải lưu giữ thông tin không gian của vân tay**.
             λ_rec được annealing từ 0 lên giá trị mục tiêu để không phá vỡ ArcFace ở giai đoạn đầu.
             **Decoder bị loại bỏ hoàn toàn khi inference** → zero deployment cost."""),
            ("L_KL", "badge-amber",
             "L_KL = KL(N(μ,σ²) ‖ N(0,I)) = ½ Σ (σ² + μ² − 1 − log σ²)",
             "Covariance regularization, tránh latent collapse",
             """Giữ phân phối posterior q(z|x) gần với prior N(0,I).
             Nếu β quá lớn → latent bị ép về Gaussian quá mạnh → mất thông tin identity.
             Dùng **KL Annealing**: β tăng dần (e.g. 0.001 → 0.05) theo schedule tuyến tính.
             Balance: KL đủ mạnh để tránh collapse, đủ nhỏ để giữ identity information."""),
            ("L_unc", "badge-pink",
             "L_unc = penalty(mean(log σ²), lower_bound, upper_bound)",
             "Heteroscedastic uncertainty calibration",
             """Kiểm soát σ²: tránh σ→0 (deterministic collapse) hoặc σ phình quá lớn.
             Khuyến khích **mẫu khó / nhiễu / mờ** sinh σ² cao hơn mẫu sạch.
             Về mặt triết lý: σ² là thước đo **aleatoric uncertainty** (bất định do data)
             — không thể giảm bằng thêm dữ liệu, chỉ có thể đo và dùng để reject."""),
        ]:
            with st.expander(f"**{name}** — {role}", expanded=(name == "L_ArcFace")):
                st.code(formula, language="text")
                st.markdown(detail)

        st.markdown("""
        > **Triết lý thiết kế:** ArcFace + Reconstruction = *gọng kìm*.
        > ArcFace **chia cắt** danh tính trên hypersphere.
        > Reconstruction **giữ lại** bản chất vật lý của đường vân tay trong không gian latent.
        """)

    # ═════════════════════════════════════════════════════════════════════════
    # Tab 3 – Training Strategy
    # ═════════════════════════════════════════════════════════════════════════
    with tabs[3]:
        st.markdown('<div class="section-header">📅 Chiến Lược Huấn Luyện: contrastive_first</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        Không train tất cả từ đầu cùng lúc. Pipeline sử dụng chiến lược <b>staged warm-up + annealing</b>
        (<code>loss_schedules=contrastive_first</code>) để tránh gradient conflict giữa ArcFace và Reconstruction.
        </div>
        """, unsafe_allow_html=True)

        stages = [
            ("🚀 Stage 1", "Stable Identity Learning",
             "Chỉ ArcFace loss, LR cao. Encoder + Projector học phân cụm identity thô. Decoder chưa được kích hoạt.",
             ["λ_arc = 1.0 (constant)", "λ_rec = 0.0", "β_KL = 0.0", "λ_unc = 0.0"]),
            ("🔀 Stage 2", "Probabilistic Head Activated",
             "Bật probabilistic sampling: z = μ + σ·ε. Thêm L_unc để σ² bắt đầu học uncertainty signal.",
             ["λ_arc = 1.0", "λ_rec = 0.0 (still off)", "β_KL = small", "λ_unc = small"]),
            ("🖼️ Stage 3", "Reconstruction Warm-up",
             "Decoder được kích hoạt dần dần (linear schedule, e.g. steps 2400→7200). λ_rec tăng từ 0 lên target để không phá ArcFace.",
             ["λ_arc = 1.0", "λ_rec: 0.0 → target (linear)", "β_KL: annealing", "λ_unc = target"]),
            ("📐 Stage 4", "KL Annealing",
             "β (KL weight) tăng từ giá trị nhỏ (e.g. 0.001) lên target (e.g. 0.05). Giúp posterior calibrate tốt mà không collapse identity.",
             ["λ_arc = 1.0", "λ_rec = target", "β_KL: 0.001 → 0.05 (linear)", "λ_unc = target"]),
        ]
        for icon, stage_name, desc, params in stages:
            with st.expander(f"{icon} **{stage_name}**"):
                sc1, sc2 = st.columns([2, 1])
                with sc1:
                    st.markdown(desc)
                with sc2:
                    for p in params:
                        color = "#6ee7b7" if "1.0" in p or "target" in p else "#64748b"
                        st.markdown(f"<div style='font-size:0.78rem; font-family:monospace; color:{color};'>{p}</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-header">📊 Loss Scheduler Types</div>', unsafe_allow_html=True)
        st.markdown("""
        | Scheduler | Dùng cho | Công thức |
        |-----------|----------|-----------|
        | **ConstantScheduler** | λ_arc (luôn = 1.0) | value = const |
        | **LinearAnnealingScheduler** | β_KL, λ_rec warm-up (theo epoch) | linear interpolation từ start→end epoch |
        | **LinearStepScheduler** | Warmup theo step (batch) | linear từ start_step → end_step |
        | **StepScheduler** | Bật loss từ epoch nhất định | = 0 trước start_epoch, = value sau |
        | **CyclicScheduler** | Optional cyclic annealing | cosine oscillation với period cố định |
        """)

    # ═════════════════════════════════════════════════════════════════════════
    # Tab 4 – Inference Modes
    # ═════════════════════════════════════════════════════════════════════════
    with tabs[4]:
        st.markdown('<div class="section-header">⚡ Bốn Chế Độ Inference</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        Tại inference, <b>decoder bị loại bỏ hoàn toàn</b>.
        Chỉ PalmEncoder (→ μ, σ) được giữ lại. Bốn chiến lược scoring được đánh giá:
        </div>
        """, unsafe_allow_html=True)

        for mode, badge, name, formula, pros, cons in [
            ("M0", "badge-violet", "Projected-mean matching",
             "s = cos(proj(μ_p), proj(μ_g))",
             ["Đơn giản nhất, không cần optimization", "Chỉ cần forward pass"],
             ["Projection head có thể làm méo khoảng cách"]),
            ("M1", "badge-cyan", "Projected-space adaptation",
             "s = cos(proj(μ_p) + δ_r, proj(μ_g))  subject to ‖δ_r‖ ≤ δ_max",
             ["Có thể cải thiện khi probe có noise nhỏ"],
             ["Cần test-time optimization", "Rủi ro: kéo unknown về known nếu không regularize"]),
            ("M2", "badge-amber", "Latent-space adaptation + KL penalty",
             "s = cos(proj(μ_p + δ_μ), proj(μ_g)) − λ·KL(N(μ_p+δ_μ, σ_p²) ‖ N(μ_p, σ_p²))",
             ["KL penalty giữ δ_μ gần phân phối ban đầu của probe"],
             ["Phức tạp hơn M1", "Cần tune λ"]),
            ("M3", "badge-emerald", "Latent-mean matching ⭐ Best",
             "s = cos(μ_p, μ_g)",
             ["Đơn giản nhất trong không gian latent", "Tốt nhất thực nghiệm", "Rank-1 cao nhất, EER thấp nhất"],
             ["Không dùng projection head → cần latent space đủ discriminative"]),
        ]:
            with st.expander(f"**{mode}** — {name}", expanded=(mode == "M3")):
                st.code(formula, language="text")
                mc1, mc2 = st.columns(2)
                mc1.markdown("**✅ Ưu điểm:**\n" + "\n".join(f"- {p}" for p in pros))
                mc2.markdown("**⚠️ Nhược điểm:**\n" + "\n".join(f"- {c}" for c in cons))

        st.markdown("""
        > **Kết luận từ paper:** M3 (cos(μₚ, μ_g)) vượt trội nhất trên tất cả backbone-dataset.
        > Projection head (ArcFace) hữu ích cho training nhưng có thể làm méo khoảng cách khi matching trực tiếp.
        > Latent mean μ mang identity signal đủ mạnh cho việc matching trực tiếp bằng cosine similarity.
        """)

    # ═════════════════════════════════════════════════════════════════════════
    # Tab 5 – Ablation Design
    # ═════════════════════════════════════════════════════════════════════════
    with tabs[5]:
        st.markdown('<div class="section-header">🔬 Thiết Kế Ablation Study</div>', unsafe_allow_html=True)

        st.markdown("**Ablation 1: Probabilistic Component**")
        st.markdown("""
        | Variant | μ | σ² | Sampling z | Decoder | Mục đích |
        |---------|---|-----|-----------|---------|---------|
        | Deterministic baseline | ✅ | ❌ | ❌ | ❌ | Không có uncertainty |
        | μ + σ² (no sampling) | ✅ | ✅ | ❌ | ❌ | Uncertainty không dùng cho decoder |
        | μ + σ² + sampling | ✅ | ✅ | ✅ | ❌ | Monte Carlo scoring |
        | VAE only | ✅ | ✅ | ✅ | ✅ | Decoder không có ArcFace |
        | **Full Proposed** | ✅ | ✅ | ✅ | ✅ | Mô hình đầy đủ |
        """)

        st.markdown("**Ablation 2: Backbone so sánh**")
        st.markdown("""
        | Backbone | Config | Đặc điểm |
        |----------|--------|---------|
        | ResNet18 | `unet_resnet` | ImageNet pretrained, deep features |
        | CCNet | `unet_ccnet` | Criss-cross attention, global context |
        | PalmNet | `unet_palmnet` | Shallow, domain-specific |
        | PalmNet-Gabor | `unet_palmnet_gabor` | Gabor-initialized, texture prior |
        """)

        st.markdown("**Ablation 3: Dataset split protocols**")
        st.markdown("""
        | Protocol | Train set | Test set | Đo gì |
        |----------|-----------|----------|-------|
        | **Ratio Split** (IITD / Tongji_Mixed) | 60% identities | 40% identities | In-domain generalization |
        | **Hand Split** | Left hand | Right hand | Cross-domain (anatomy) |
        | **Session Split** (Tongji) | Session 1 | Session 2 | Temporal drift robustness |
        """)

        st.markdown("**Ablation 4: Inference Mode comparison**")
        st.markdown("""
        | Mode | Không gian | Adaptation | KL Gate |
        |------|-----------|-----------|---------|
        | M0 | Projected | ❌ | ❌ |
        | M1 | Projected | δ_r ✅ | ❌ |
        | M2 | Latent→Projected | δ_μ ✅ | ✅ |
        | M3 | Latent | ❌ | ❌ |
        | M4 | Latent | δ_μ ✅ | ✅ |
        """)

    # ═════════════════════════════════════════════════════════════════════════
    # Tab 6 – Key Results
    # ═════════════════════════════════════════════════════════════════════════
    with tabs[6]:
        st.markdown('<div class="section-header">📊 Kết Quả Thực Nghiệm</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        Tóm tắt kết quả từ file <code>tasks/summary_table.csv</code> và <code>experiments/report/experiment_results.md</code>.
        Kết quả đầy đủ xem tại trang <b>📊 Results Overview</b>.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Kết quả nổi bật (Proposed model – Mode M3):**")
        st.markdown("""
        | Model | Dataset | Open Rank-1 | Open EER | AUROC | OSCR |
        |-------|---------|------------|---------|-------|------|
        | **CCNet** | Own + IITD + Tongji | **99.4%** | **0.73%** | **99.5%** | **99.3%** |
        | ResNet18 | IITD | 98.3% | 2.61% | 99.5% | 98.2% |
        | ResNet18 | Tongji | 89.5% | 5.05% | 92.7% | 86.7% |
        | PalmNet | Own + IITD + Tongji | 92.6% | 3.66% | — | — |
        """)

        st.markdown("**So sánh Baseline vs. Proposed (ResNet, Mode M3, IITD→IITD):**")
        st.markdown("""
        | | Baseline (M3) | Proposed (M3) | Δ |
        |--|---------------|---------------|---|
        | Open Rank-1 | 92.8% | **98.3%** | +5.5% |
        | Open EER | 7.4% | **2.6%** | −4.8% |
        | AUROC | 95.0% | **99.5%** | +4.5% |
        | DIR@1% | 86.2% | **95.8%** | +9.6% |
        """)

        st.markdown("""
        > **Nhận xét từ báo cáo thực nghiệm:**
        > - Trên OwnDataset: ResNet+ArcFace đạt Rank-1 > 99% ngay cả ở baseline.
        > - Trên IITD & Tongji (cross-domain): Proposed model cải thiện đáng kể so với baseline.
        > - Optimize-r (M1/M2) giúp nhiều nhất trên Tongji (Rank-1 từ 33% → 84%).
        > - M3 (latent-mean) ổn định nhất và đạt kết quả tốt nhất trên hầu hết settings.
        """)

    # ═════════════════════════════════════════════════════════════════════════
    # Tab 7 – Open-Set Rejection Rule
    # ═════════════════════════════════════════════════════════════════════════
    with tabs[7]:
        st.markdown('<div class="section-header">🚦 Quy Tắc Từ Chối Open-Set</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        Probe <b>p</b> được chấp nhận là identity <b>g*</b> chỉ khi thỏa mãn <b>đồng thời tất cả</b>
        ba điều kiện dưới đây. Các ngưỡng τ được hiệu chỉnh <b>cùng nhau</b> trên validation split
        — không bao giờ tune trực tiếp trên test set.
        </div>
        """, unsafe_allow_html=True)

        rc = st.columns(3)
        for col, (icon, title, formula, detail, variant) in zip(rc, [
            ("⚠️", "Uncertainty Gate", "U_p ≤ τ_U",
             """**U_p** là tổng uncertainty của probe:
             ```
             U_p = mean(σ²_p)  hoặc  sum(log σ²_p)
             ```
             Nếu σ² cao → ảnh mờ / nhiễu / ROI lệch / stranger → **reject**.
             Threshold τ_U được chọn trên validation để balance FRR.""",
             ""),
            ("🎯", "Similarity Gate", "S(p, g*) ≥ τ_S",
             """**S(p, g*)** là cosine similarity giữa probe và gallery:
             - M3: `cos(μ_p, μ_{g*})`
             - M0: `cos(proj(μ_p), proj(μ_{g*}))`

             Threshold τ_S phải đủ cao để loại impostor.
             Kết hợp với τ_U tạo two-gate rejection.""",
             "warm"),
            ("📐", "KL Divergence Gate", "D_SKL(p, g*) ≤ τ_K",
             """**D_SKL** là symmetric KL divergence giữa hai Gaussian:
             ```
             D_SKL(p‖g) = ½[KL(p‖g) + KL(g‖p)]
             ```
             Đo khoảng cách phân phối — không chỉ khoảng cách điểm.
             Stranger thường có D_SKL lớn hơn genuine pair.""",
             "cool"),
        ]):
            col.markdown(f"""
            <div class="metric-card {variant}" style="padding:1.2rem;">
                <div style="font-size:1.8rem; margin-bottom:0.5rem;">{icon}</div>
                <div style="font-weight:700; margin-bottom:0.3rem;">{title}</div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:1.1rem; color:var(--text-primary); margin-bottom:0.5rem;">{formula}</div>
            </div>
            """, unsafe_allow_html=True)
            col.markdown(detail)

        st.markdown('<div class="section-header">🔄 Decision Flow</div>', unsafe_allow_html=True)
        st.code("""
Input probe x_p
    ↓
Encoder → μ_p, σ_p
    ↓
Quality / Uncertainty check:
    if U_p > τ_U  →  REJECT (uncertain / low quality image)
    ↓
Retrieve top-K gallery candidates by cos(μ_p, μ_g)
    ↓
For best candidate g*:
    if S(p, g*) < τ_S  →  REJECT (too different from all gallery)
    if D_SKL(p, g*) > τ_K  →  REJECT (distributions incompatible)
    ↓
ACCEPT as identity g*
        """, language="text")

        st.markdown("""
        > **Thiết kế 3-gate** kết hợp ba tín hiệu khác nhau:
        > uncertainty (chất lượng ảnh), similarity (khoảng cách đặc trưng), và KL (tương thích phân phối).
        > Mỗi gate bắt được loại rejection khác nhau, tạo defense-in-depth cho hệ thống biometric.
        """)

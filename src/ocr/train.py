import torch


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    for i, (pixel_values, labels, _, _) in enumerate(loader):
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)

        outputs = model(
            pixel_values=pixel_values,
            labels=labels
        )

        loss = outputs.loss

        loss.backward()

        # 🔥 stabilize training
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        if i % 10 == 0:
            print(f"➡ Batch {i} | Loss: {loss.item():.4f}")

    return total_loss


def predict_sample(model, processor, image, device):
    model.eval()

    pixel_values = processor(
        images=image,
        return_tensors="pt"
    ).pixel_values.to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values,

            # 🔥 KEY FIXES (reduce hallucination)
            num_beams=1,              # was 5 → now 1 (less language bias)
            do_sample=False,
            early_stopping=True,
            no_repeat_ngram_size=2,
            max_length=64             # shorter = less garbage text
        )

    return processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]
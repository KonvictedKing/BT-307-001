import os
import re
import json
import time
import urllib.parse
from datetime import datetime
from io import BytesIO

import feedparser
import requests
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from groq import Groq
from google import genai

PAGE_ID = os.environ.get("FB_PAGE_ID")
IG_USER_ID = os.environ.get("IG_USER_ID")
ACCESS_TOKEN = os.environ.get("FB_ACCESS_TOKEN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

ALL_FEEDS = [
    {"name": "The Daily Star", "url": "https://www.thedailystar.net/frontpage/rss.xml"},
    {"name": "Prothom Alo", "url": "https://www.prothomalo.com/feed"},
    {"name": "The Business Standard", "url": "https://www.tbsnews.net/rss.xml"},
    {"name": "Samakal", "url": "https://www.samakal.com/feed"},
    {"name": "BBC Bangla", "url": "https://feeds.bbci.co.uk/bengali/rss.xml"},
    {"name": "Kaler Kantho", "url": "https://www.kalerkantho.com/rss.xml"},
    {"name": "Daily Ittefaq", "url": "https://www.ittefaq.com.bd/rss.xml"},
    {"name": "Daily Jugantor", "url": "https://www.jugantor.com/feed"},
    {"name": "Dhaka Tribune", "url": "https://www.dhakatribune.com/feed"},
    {"name": "BDNews24", "url": "https://bangla.bdnews24.com/rss.xml"},
    {"name": "Banglanews24", "url": "https://www.banglanews24.com/rss/rss.xml"},
    {"name": "Jago News", "url": "https://www.jagonews24.com/rss/rss.xml"},
    {"name": "Dhaka Post", "url": "https://www.dhakapost.com/rss.xml"},
    {"name": "Kalbela", "url": "https://www.kalbela.com/feed"},
    {"name": "Somoy TV", "url": "https://www.somoynews.tv/rss.xml"},
    {"name": "Channel 24", "url": "https://www.channel24bd.tv/rss.xml"},
    {"name": "Jamuna TV", "url": "https://www.jamuna.tv/feed"}
]

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "bn,en-US;q=0.9,en;q=0.8"
}

POLITICS_KEYWORDS = [
    "সরকার", "রাজনৈতিক", "নির্বাচন", "উপদেষ্টা", "আওয়ামী", "বিএনপি", "জামায়াত", "সংসদ", "আইন", 
    "আদালত", "মামলা", "গ্রেফতার", "পুলিশ", "সেনাবাহিনী", "রিমান্ড", "politics", "political", 
    "election", "government", "adviser", "bnp", "awami", "court", "arrest", "minister", "parliament"
]

def load_state():
    state = {"posted_urls": []}
    if os.path.exists("posted_urls.json"):
        try:
            with open("posted_urls.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    state["posted_urls"] = data
                elif isinstance(data, dict):
                    state["posted_urls"] = data.get("posted_urls", [])
        except Exception:
            pass
    return state

def save_state(state):
    state["posted_urls"] = state["posted_urls"][-500:]
    with open("posted_urls.json", "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

def pre_clean_text(text):
    cleaned = re.sub(r"<[^>]+>", "", text)
    patterns = [r"\[.*?\]", r"\(.*?\)", r"\|.*$"]
    for p in patterns:
        cleaned = re.sub(p, "", cleaned)
    cleaned = re.sub(r"Photo:.*?(\.|$)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"Author:.*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()

def get_live_groq_models():
    if not groq_client:
        return []
    try:
        available = [m.id for m in groq_client.models.list().data if "whisper" not in m.id.lower() and "guard" not in m.id.lower()]
        priority = ["qwen/qwen3.6-27b", "openai/gpt-oss-20b", "openai/gpt-oss-120b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
        sorted_models = [m for m in priority if m in available] + [m for m in available if m not in priority]
        return sorted_models
    except Exception as e:
        print(f"Could not list Groq models: {e}", flush=True)
        return []

ACTIVE_GROQ_MODELS = get_live_groq_models()

def strip_all_thinking(text):
    """Purges reasoning tokens and internal thought dumps."""
    if not text:
        return ""
    # Strip matched <think>...</think>
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    # Strip unclosed <think> if response truncated
    text = re.sub(r"<think>[\s\S]*", "", text, flags=re.IGNORECASE)
    # Strip English analytical intro phrases
    text = re.sub(r"^(?:Here['’]s a thinking process|Analyze User Input|Translate & Adapt Content)[\s\S]*?\n\n", "", text, flags=re.IGNORECASE)
    return text.strip()

def query_llm_dual_engine(prompt):
    if groq_client and ACTIVE_GROQ_MODELS:
        for model in ACTIVE_GROQ_MODELS[:3]:
            try:
                res = groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are the Senior Bangla Editor of Bongo Tribune. Do NOT think out loud. Output ONLY the requested Bengali format immediately."},
                        {"role": "user", "content": prompt}
                    ],
                    model=model,
                    temperature=0.1,
                    max_tokens=350,
                )
                raw_text = res.choices[0].message.content
                clean = strip_all_thinking(raw_text)
                if len(clean) > 20:
                    return clean
            except Exception as ge:
                print(f"Groq {model} error: {ge}", flush=True)

    if gemini_client:
        for model in ["gemini-3.6-flash", "gemini-3.5-flash"]:
            for attempt in range(2):
                try:
                    res = gemini_client.models.generate_content(
                        model=model,
                        contents=prompt,
                    )
                    if res and res.text:
                        clean = strip_all_thinking(res.text)
                        if len(clean) > 20:
                            return clean
                except Exception as gme:
                    err_msg = str(gme)
                    if "429" in err_msg and attempt == 0:
                        time.sleep(15)
                    else:
                        break
    return None

def analyze_and_score_news(raw_title, raw_summary, source_name):
    clean_t = pre_clean_text(raw_title)
    clean_s = pre_clean_text(raw_summary) if raw_summary else clean_t

    prompt = f"""News Source: {source_name}
Title: {clean_t}
Summary: {clean_s}

Write a Bengali news card for Bongo Tribune.
CRITICAL INSTRUCTIONS:
1. Output MUST be 100% in fluent journalistic Bengali (বাংলা). No English.
2. Provide a 3 to 4 sentence rich Bengali summary (40-60 words).
3. Do not include thinking or notes.

Format strictly as:
HEADLINE: <বাংলায় মূল শিরোনাম>
SUB_HEADLINE: <বাংলায় উপ-শিরোনাম অথবা None>
SUMMARY: <বাংলায় ৩-৪ বাক্যের বিস্তারিত প্রতিবেদন>
IS_POLITICS: <YES or NO>
ENGAGEMENT_SCORE: <1-10>"""

    response_text = query_llm_dual_engine(prompt)
    if not response_text:
        return None

    try:
        clean_resp = strip_all_thinking(response_text)
        clean_resp = re.sub(r"[*#_`]", "", clean_resp)

        headline = ""
        sub_headline = ""
        summary = ""

        hl_m = re.search(r"HEADLINE:\s*(.*?)(?=\n(?:SUB_HEADLINE|SUMMARY|IS_POLITICS)|$)", clean_resp, re.IGNORECASE)
        if hl_m:
            headline = hl_m.group(1).strip(' \n"')

        sub_m = re.search(r"SUB_HEADLINE:\s*(.*?)(?=\n(?:SUMMARY|IS_POLITICS|ENGAGEMENT)|$)", clean_resp, re.IGNORECASE)
        if sub_m:
            cand = sub_m.group(1).strip(' \n"')
            if cand.lower() != "none" and len(cand) > 3:
                sub_headline = cand

        sum_m = re.search(r"SUMMARY:\s*(.*?)(?=\n(?:IS_POLITICS|ENGAGEMENT_SCORE)|$)", clean_resp, re.DOTALL | re.IGNORECASE)
        if sum_m:
            summary = sum_m.group(1).strip(' \n"')

        # Fallback line scan
        if not headline or not summary:
            for line in clean_resp.split("\n"):
                line = line.strip()
                if re.match(r"^HEADLINE:\s*", line, re.IGNORECASE) and not headline:
                    headline = re.sub(r"^HEADLINE:\s*", "", line, flags=re.IGNORECASE).strip(' "')
                elif re.match(r"^SUMMARY:\s*", line, re.IGNORECASE) and not summary:
                    summary = re.sub(r"^SUMMARY:\s*", "", line, flags=re.IGNORECASE).strip(' "')

        # Guarantee non-empty
        if not headline or len(headline) < 5 or "<বাংলায়" in headline or "think>" in headline:
            headline = clean_t
        if not summary or len(summary) < 15 or "<বাংলায়" in summary or "think>" in summary:
            summary = clean_s[:260]

        pol_m = re.search(r"IS_POLITICS:\s*(YES|NO)", clean_resp, re.IGNORECASE)
        is_pol = bool(pol_m and "YES" in pol_m.group(1).upper())
        if not is_pol:
            is_pol = any(k in (clean_t + " " + clean_s).lower() for k in POLITICS_KEYWORDS)

        score_match = re.search(r"ENGAGEMENT_SCORE:\s*(\d+)", clean_resp, re.IGNORECASE)
        score = int(score_match.group(1)) if score_match else 6

        return {
            "headline": headline,
            "sub_headline": sub_headline,
            "summary": summary,
            "is_politics": is_pol,
            "score": score
        }
    except Exception as e:
        print(f"Parsing error: {e}", flush=True)
        return None

def clean_and_maximize_image_url(url):
    if not url:
        return url
    url = re.sub(r'([?&])(w|width|h|height|max_width|resize|crop)=\d+[^&]*', '', url)
    url = url.replace('/thumb/', '/').replace('/300/', '/1200/').replace('/640/', '/1200/')
    url = re.sub(r'\?&+', '?', url)
    url = re.sub(r'&+', '&', url)
    url = re.sub(r'[?&]$', '', url)
    return url

def search_related_news_image(query):
    try:
        clean_q = re.sub(r"[^\w\s]", " ", query)[:45].strip()
        encoded = urllib.parse.quote(f"{clean_q} news bangladesh")
        url = f"https://html.duckduckgo.com/html/?q={encoded}"
        resp = requests.get(url, headers=BROWSER_HEADERS, timeout=6)
        candidates = re.findall(r'//external-content\.duckduckgo\.com/iu/\?u=(https?://[^&"\']+)', resp.text)
        for cand in candidates:
            dec = urllib.parse.unquote(cand)
            if dec.endswith(('.jpg', '.jpeg', '.png', '.webp')) and 'logo' not in dec.lower() and 'icon' not in dec.lower():
                return dec
    except Exception:
        pass
    return None

def extract_high_res_image(entry):
    if 'media_content' in entry and len(entry.media_content) > 0:
        url = entry.media_content[0].get('url')
        if url and not url.endswith(('.svg', '.gif')):
            return clean_and_maximize_image_url(url)

    if 'enclosures' in entry and len(entry.enclosures) > 0:
        url = entry.enclosures[0].get('href')
        if url and not url.endswith(('.svg', '.gif')):
            return clean_and_maximize_image_url(url)

    for field in ['summary', 'description']:
        text_val = entry.get(field, "")
        img_m = re.search(r'<img[^>]+src=["\'](https?://[^"\']+)["\']', text_val, re.IGNORECASE)
        if img_m:
            cand = img_m.group(1)
            if not cand.endswith(('.svg', '.gif', '.ico')) and 'avatar' not in cand.lower():
                return clean_and_maximize_image_url(cand)

    try:
        resp = requests.get(entry.link, timeout=7, headers=BROWSER_HEADERS)
        if resp.status_code == 200:
            html = resp.text
            patterns = [
                r'<meta[^>]+property=["\']og:image:secure_url["\'][^>]+content=["\'](https?://[^"\']+)["\']',
                r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\'](https?://[^"\']+)["\']',
                r'<meta[^>]+content=["\'](https?://[^"\']+)["\'][^>]+property=["\']og:image["\']',
                r'<meta[^>]+name=["\']twitter:image["\'][^>]+content=["\'](https?://[^"\']+)["\']',
                r'<link[^>]+rel=["\']image_src["\'][^>]+href=["\'](https?://[^"\']+)["\']',
                r'<source[^>]+srcset=["\'](https?://[^"\', ]+)["\']',
                r'<figure[^>]*>.*?<img[^>]+src=["\'](https?://[^"\']+)["\']'
            ]
            for pat in patterns:
                m = re.search(pat, html, re.DOTALL | re.IGNORECASE)
                if m:
                    candidate = m.group(1)
                    if not candidate.endswith(('.svg', '.gif', '.ico')) and 'avatar' not in candidate.lower() and 'logo' not in candidate.lower():
                        return clean_and_maximize_image_url(candidate)
    except Exception:
        pass

    return search_related_news_image(entry.title)

def ensure_font_downloaded():
    font_file = "HindSiliguri-Bold.ttf"
    if not os.path.exists(font_file):
        url = "https://raw.githubusercontent.com/google/fonts/main/ofl/hindsiliguri/HindSiliguri-Bold.ttf"
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                with open(font_file, "wb") as f:
                    f.write(r.content)
        except Exception:
            pass
    return font_file if os.path.exists(font_file) else None

def get_font(size=32):
    local_font = ensure_font_downloaded()
    if local_font:
        try:
            return ImageFont.truetype(local_font, size)
        except Exception:
            pass
    fallbacks = [
        "/usr/share/fonts/truetype/noto/NotoSansBengali-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    ]
    for fb in fallbacks:
        if os.path.exists(fb):
            try:
                return ImageFont.truetype(fb, size)
            except Exception:
                continue
    return ImageFont.load_default()

def wrap_text(text, font, max_width, draw):
    lines = []
    words = text.split()
    current_line = []
    for word in words:
        test_line = " ".join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if (bbox[2] - bbox[0]) <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
    if current_line:
        lines.append(" ".join(current_line))
    return lines

def get_asset(base_name):
    candidates = [
        f"{base_name}.png", f"{base_name}.jpg", f"{base_name}.jpeg",
        f"{base_name}.png.png", f"{base_name}.jpg.jpg", f"{base_name}.jpeg.jpeg"
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def download_image_robust(url, fallback_query=""):
    def try_fetch(target_url):
        if not target_url:
            return None
        try:
            resp = requests.get(target_url, timeout=10, headers=BROWSER_HEADERS)
            if resp.status_code == 200 and len(resp.content) > 3000:
                img = Image.open(BytesIO(resp.content)).convert("RGB")
                if img.width > 200 and img.height > 150:
                    return img
        except Exception:
            pass
        return None

    img = try_fetch(url)
    if img:
        return img

    if fallback_query:
        fallback_url = search_related_news_image(fallback_query)
        img = try_fetch(fallback_url)
        if img:
            return img

    return None

def build_bongo_card(image_url, headline, sub_headline, summary, source_name, is_square=False):
    width = 1080
    height = 1080 if is_square else 1350
    top_h = int(height * 0.60)

    # Base card: Lower 40% is solid maroon (#4c0000)
    card = Image.new("RGB", (width, height), color="#4c0000")

    # 1. TOP 60% PHOTO WITH GAUSSIAN BLUR FILLER
    raw_img = download_image_robust(image_url, fallback_query=headline)
    if raw_img:
        try:
            bg_blur = raw_img.resize((width, top_h), Image.Resampling.BILINEAR)
            bg_blur = bg_blur.filter(ImageFilter.GaussianBlur(radius=32))

            scale = max(width / raw_img.width, top_h / raw_img.height)
            new_w = max(1, int(raw_img.width * scale))
            new_h = max(1, int(raw_img.height * scale))
            fit_img = raw_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            paste_x = (width - new_w) // 2
            paste_y = (top_h - new_h) // 2
            bg_blur.paste(fit_img, (paste_x, paste_y))
            card.paste(bg_blur, (0, 0))
        except Exception as e:
            print(f"Photo render note: {e}", flush=True)

    # 2. TOP HEADER LOGO ("Bongo Tribune")
    header_path = get_asset("header_logo")
    if header_path:
        try:
            h_logo = Image.open(header_path).convert("RGBA")
            aspect = h_logo.width / h_logo.height
            logo_h = 75
            h_logo = h_logo.resize((int(logo_h * aspect), logo_h), Image.Resampling.LANCZOS)
            card.paste(h_logo, (50, 45), mask=h_logo.split()[3])
        except Exception:
            pass

    # 3. EXPANDED CHAT BUBBLE
    bubble_w = 930
    bubble_h = 570 if is_square else 660
    bubble_x = (width - bubble_w) // 2
    footer_margin = 85
    bubble_y = height - bubble_h - footer_margin

    bubble_img = Image.new("RGBA", (bubble_w, bubble_h), (0, 0, 0, 0))
    b_draw = ImageDraw.Draw(bubble_img)
    tail_h = 45

    # White Chat Bubble Background
    b_draw.rounded_rectangle([(0, 0), (bubble_w, bubble_h - tail_h)], radius=26, fill=(255, 255, 255, 255))

    # Speech Tail pointing down toward the right
    tail = [(bubble_w - 150, bubble_h - tail_h), (bubble_w - 55, bubble_h), (bubble_w - 55, bubble_h - tail_h)]
    b_draw.polygon(tail, fill=(255, 255, 255, 255))

    # WATERMARK SEAL: Subtle 10% alpha background tint
    watermark_path = get_asset("watermark")
    if watermark_path:
        try:
            wm = Image.open(watermark_path).convert("RGBA")
            wm_size = int(bubble_h * 0.65)
            wm = wm.resize((wm_size, wm_size), Image.Resampling.LANCZOS)
            
            r, g, b, a = wm.split()
            subtle_alpha = a.point(lambda p: int(p * 0.10))
            tinted_wm = Image.merge("RGBA", (r, g, b, subtle_alpha))

            wm_x = (bubble_w - wm_size) // 2
            wm_y = (bubble_h - tail_h - wm_size) // 2
            bubble_img.paste(tinted_wm, (wm_x, wm_y), mask=tinted_wm)
        except Exception as e:
            print(f"Watermark note: {e}", flush=True)

    font_hl = get_font(44 if not is_square else 36)
    font_sub = get_font(30 if not is_square else 25)
    font_sum = get_font(26 if not is_square else 22)

    text_pad_x = 45
    inner_w = bubble_w - (text_pad_x * 2)

    hl_lines = wrap_text(headline, font_hl, inner_w, b_draw)[:3]
    sub_lines = wrap_text(sub_headline, font_sub, inner_w, b_draw)[:2] if sub_headline else []
    sum_lines = wrap_text(summary, font_sum, inner_w, b_draw)[:5]

    spacing_hl = 12
    spacing_sub = 10
    spacing_sum = 8

    total_text_h = 0
    for l in hl_lines:
        bb = b_draw.textbbox((0, 0), l, font=font_hl)
        total_text_h += (bb[3] - bb[0]) + spacing_hl
    if sub_lines:
        total_text_h += 6
        for l in sub_lines:
            bb = b_draw.textbbox((0, 0), l, font=font_sub)
            total_text_h += (bb[3] - bb[0]) + spacing_sub
    if sum_lines:
        total_text_h += 16
        for l in sum_lines:
            bb = b_draw.textbbox((0, 0), l, font=font_sum)
            total_text_h += (bb[3] - bb[0]) + spacing_sum

    usable_bubble_h = bubble_h - tail_h
    start_y = max(35, (usable_bubble_h - total_text_h) // 2)

    # Render Headline (Maroon, Centered)
    cur_y = start_y
    for line in hl_lines:
        bbox = b_draw.textbbox((0, 0), line, font=font_hl)
        line_w = bbox[2] - bbox[0]
        b_draw.text((text_pad_x + (inner_w - line_w) // 2, cur_y), line, fill="#4c0000", font=font_hl)
        cur_y += (bbox[3] - bbox[0]) + spacing_hl

    # Render Sub-headline (Maroon, Centered)
    if sub_lines:
        cur_y += 6
        for line in sub_lines:
            bbox = b_draw.textbbox((0, 0), line, font=font_sub)
            line_w = bbox[2] - bbox[0]
            b_draw.text((text_pad_x + (inner_w - line_w) // 2, cur_y), line, fill="#4c0000", font=font_sub)
            cur_y += (bbox[3] - bbox[0]) + spacing_sub

    # Render Summary (Pure Black, Centered)
    if sum_lines:
        cur_y += 16
        for line in sum_lines:
            bbox = b_draw.textbbox((0, 0), line, font=font_sum)
            line_w = bbox[2] - bbox[0]
            b_draw.text((text_pad_x + (inner_w - line_w) // 2, cur_y), line, fill="#000000", font=font_sum)
            cur_y += (bbox[3] - bbox[0]) + spacing_sum

    card.paste(bubble_img, (bubble_x, bubble_y), mask=bubble_img)

    # 4. FOOTER: DATE & SOURCE ALIGNMENT
    draw = ImageDraw.Draw(card)
    font_footer = get_font(26)
    date_str = datetime.utcnow().strftime("%d %B").upper()
    footer_y = height - 60

    draw.text((60, footer_y), date_str, fill="#ffffff", font=font_footer)

    clean_source = source_name.replace("http://", "").replace("https://", "").replace("www.", "")
    source_str = f"Source : {clean_source}"
    src_bbox = draw.textbbox((0, 0), source_str, font=font_footer)
    src_w = src_bbox[2] - src_bbox[0]
    draw.text((width - 60 - src_w, footer_y), source_str, fill="#ffffff", font=font_footer)

    filename = "final_card_ig.jpg" if is_square else "final_card_fb.jpg"
    card.save(filename, "JPEG", quality=95)
    return filename

def build_instagram_story(feed_card_path):
    story_w, story_h = 1080, 1920
    feed_card = Image.open(feed_card_path).convert("RGB")
    bg = feed_card.resize((story_w, story_h), Image.Resampling.BILINEAR)
    bg = bg.filter(ImageFilter.GaussianBlur(radius=45))
    dark_overlay = Image.new("RGB", (story_w, story_h), color="#000000")
    bg = Image.blend(bg, dark_overlay, alpha=0.35)

    target_card_w = int(story_w * 0.88)
    scaled_card = feed_card.resize((target_card_w, target_card_w), Image.Resampling.LANCZOS)

    radius = 32
    mask = Image.new("L", (target_card_w, target_card_w), 0)
    draw_mask = ImageDraw.Draw(mask)
    draw_mask.rounded_rectangle([(0, 0), (target_card_w, target_card_w)], radius=radius, fill=255)

    card_x = (story_w - target_card_w) // 2
    card_y = (story_h - target_card_w) // 2 - 50
    bg.paste(scaled_card, (card_x, card_y), mask)

    draw = ImageDraw.Draw(bg)
    handle_font = get_font(34)
    draw.text((card_x + 10, card_y + target_card_w + 30), "@bongo.tribune", fill="#ffffff", font=handle_font)

    story_path = "final_story_ig.jpg"
    bg.save(story_path, "JPEG", quality=95)
    return story_path

def get_fb_image_url(photo_id):
    try:
        url = f"https://graph.facebook.com/v20.0/{photo_id}?fields=images&access_token={ACCESS_TOKEN}"
        res = requests.get(url, timeout=10).json()
        if "images" in res and len(res["images"]) > 0:
            return res["images"][0]["source"]
    except Exception:
        pass
    return None

def post_facebook_feed(image_path, caption):
    url = f"https://graph.facebook.com/v20.0/{PAGE_ID}/photos"
    payload = {"caption": caption, "published": "true", "access_token": ACCESS_TOKEN}
    with open(image_path, "rb") as f:
        res = requests.post(url, files={"source": f}, data=payload).json()
        print("FB Feed response:", res, flush=True)
        return res

def post_facebook_comment(target_id, message):
    url = f"https://graph.facebook.com/v20.0/{target_id}/comments"
    payload = {"message": message, "access_token": ACCESS_TOKEN}
    try:
        res = requests.post(url, data=payload, timeout=15).json()
        print("FB Comment response:", res, flush=True)
        return res
    except Exception as e:
        print(f"FB Comment error: {e}", flush=True)
        return None

def post_facebook_story(image_path):
    url = f"https://graph.facebook.com/v20.0/{PAGE_ID}/photos"
    payload = {"published": "false", "temporary": "true", "access_token": ACCESS_TOKEN}
    with open(image_path, "rb") as f:
        res = requests.post(url, files={"source": f}, data=payload).json()
        photo_id = res.get("id")
        if photo_id:
            story_url = f"https://graph.facebook.com/v20.0/{PAGE_ID}/photo_stories"
            requests.post(story_url, data={"photo_id": photo_id, "access_token": ACCESS_TOKEN})

def wait_for_ig_container(creation_id):
    status_url = f"https://graph.facebook.com/v20.0/{creation_id}?fields=status_code&access_token={ACCESS_TOKEN}"
    for _ in range(12):
        time.sleep(5)
        res = requests.get(status_url).json()
        if res.get("status_code") == "FINISHED":
            return True
        if res.get("status_code") == "ERROR":
            print(f"IG container error: {res}", flush=True)
            return False
    return True

def post_instagram_feed(image_url, caption):
    if not IG_USER_ID:
        return None
    create_url = f"https://graph.facebook.com/v20.0/{IG_USER_ID}/media"
    payload = {"image_url": image_url, "caption": caption, "access_token": ACCESS_TOKEN}
    res = requests.post(create_url, data=payload).json()
    creation_id = res.get("id")
    if not creation_id:
        print(f"IG container create error: {res}", flush=True)
        return None

    if wait_for_ig_container(creation_id):
        pub_url = f"https://graph.facebook.com/v20.0/{IG_USER_ID}/media_publish"
        pub_res = requests.post(pub_url, data={"creation_id": creation_id, "access_token": ACCESS_TOKEN}).json()
        print("IG Feed published:", pub_res, flush=True)
        return pub_res.get("id")
    return None

def post_instagram_story(story_image_url):
    if not IG_USER_ID:
        return
    create_url = f"https://graph.facebook.com/v20.0/{IG_USER_ID}/media"
    payload = {"image_url": story_image_url, "media_type": "STORIES", "access_token": ACCESS_TOKEN}
    res = requests.post(create_url, data=payload).json()
    creation_id = res.get("id")
    if not creation_id:
        return
    if wait_for_ig_container(creation_id):
        pub_url = f"https://graph.facebook.com/v20.0/{IG_USER_ID}/media_publish"
        requests.post(pub_url, data={"creation_id": creation_id, "access_token": ACCESS_TOKEN})

def publish_article(entry, source_name, img_url, curated):
    headline = curated["headline"]
    sub_headline = curated["sub_headline"]
    summary = curated["summary"]

    print(f"Publishing article: {headline}", flush=True)

    fb_card_path = build_bongo_card(img_url, headline, sub_headline, summary, source_name, is_square=False)
    if not fb_card_path:
        return False

    ig_card_path = build_bongo_card(img_url, headline, sub_headline, summary, source_name, is_square=True)

    # Facebook Caption: Headline + Summary + First Comment Prompt
    post_caption_fb = f"{headline}\n\n{summary}\n\n(বিস্তারিত প্রথম কমেন্টে)"
    comment_text_fb = f"সম্পূর্ণ প্রতিবেদনটি পড়তে ভিজিট করুন:\n{entry.link}"

    fb_res = post_facebook_feed(fb_card_path, post_caption_fb)
    fb_photo_id = fb_res.get("id") if isinstance(fb_res, dict) else None

    if not fb_photo_id:
        print(f"Facebook upload rejected: {fb_res}", flush=True)
        return False

    post_facebook_comment(fb_photo_id, comment_text_fb)

    try:
        post_facebook_story(fb_card_path)
    except Exception as err:
        print(f"FB Story bypass: {err}", flush=True)

    if IG_USER_ID:
        try:
            temp_res = requests.post(
                f"https://graph.facebook.com/v20.0/{PAGE_ID}/photos",
                files={"source": open(ig_card_path, "rb")},
                data={"published": "false", "temporary": "true", "access_token": ACCESS_TOKEN}
            ).json()
            ig_cdn_url = get_fb_image_url(temp_res.get("id"))

            if ig_cdn_url:
                # INSTAGRAM CAPTION: Headline + Source + Hashtags (SUMMARY SKIPPED AS REQUESTED)
                ig_caption = f"{headline}\n\nসূত্র: {source_name}\n\n#bongotribune #banglanews #bangladesh #news"
                post_instagram_feed(ig_cdn_url, ig_caption)

            styled_story_path = build_instagram_story(ig_card_path)
            story_res = requests.post(
                f"https://graph.facebook.com/v20.0/{PAGE_ID}/photos",
                files={"source": open(styled_story_path, "rb")},
                data={"published": "false", "temporary": "true", "access_token": ACCESS_TOKEN}
            ).json()
            story_cdn = get_fb_image_url(story_res.get("id"))

            if story_cdn:
                post_instagram_story(story_cdn)
        except Exception as err:
            print(f"Instagram error: {err}", flush=True)

    return True

def scan_feeds_smart(state):
    print(f"Smart-scanning {len(ALL_FEEDS)} media feeds with Dual-AI engine...", flush=True)
    qualifying_candidates = []
    prefiltered_entries = []

    feed_entries_map = {}
    for feed in ALL_FEEDS:
        try:
            parsed = feedparser.parse(feed["url"])
            valid_for_this_feed = []

            for entry in parsed.entries[:6]:
                if entry.link in state["posted_urls"]:
                    continue
                clean_t = pre_clean_text(entry.title)
                if len(clean_t.split()) < 3:
                    continue

                raw_summary = entry.get("summary", "")
                is_pol_hint = any(k in (clean_t + " " + raw_summary).lower() for k in POLITICS_KEYWORDS)
                valid_for_this_feed.append({
                    "entry": entry,
                    "source": feed["name"],
                    "is_pol_hint": is_pol_hint
                })
            if valid_for_this_feed:
                feed_entries_map[feed["name"]] = valid_for_this_feed
        except Exception:
            continue

    max_depth = max([len(v) for v in feed_entries_map.values()]) if feed_entries_map else 0
    for depth in range(min(max_depth, 2)):
        for source_name, entries in feed_entries_map.items():
            if depth < len(entries):
                prefiltered_entries.append(entries[depth])

    seen_sources = set()
    evaluation_queue = []
    for e in prefiltered_entries:
        if e["source"] not in seen_sources:
            seen_sources.add(e["source"])
            evaluation_queue.append(e)
        if len(evaluation_queue) >= 10:
            break

    for item in evaluation_queue:
        entry = item["entry"]
        curated = analyze_and_score_news(entry.title, entry.get("summary", ""), item["source"])
        if curated:
            img_url = extract_high_res_image(entry)
            qualifying_candidates.append({
                "entry": entry,
                "source_name": item["source"],
                "img_url": img_url,
                "curated": curated,
                "is_politics": curated["is_politics"],
                "score": curated["score"]
            })
            print(f"Evaluated: [{item['source']}] {curated['headline']}", flush=True)
        time.sleep(1)

    qualifying_candidates.sort(key=lambda x: x["score"], reverse=True)
    return qualifying_candidates

def main():
    state = load_state()
    candidates = scan_feeds_smart(state)

    if not candidates:
        print("No qualifying Bangladesh articles evaluated in this run.", flush=True)
        return

    published_count = 0
    used_sources = set()

    # Slot 1: Top Politics Story
    pol_candidates = [c for c in candidates if c["is_politics"]]
    if pol_candidates:
        chosen_pol = pol_candidates[0]
        print(f"Publishing Slot 1 (Politics from {chosen_pol['source_name']})...", flush=True)
        if publish_article(chosen_pol["entry"], chosen_pol["source_name"], chosen_pol["img_url"], chosen_pol["curated"]):
            state["posted_urls"].append(chosen_pol["entry"].link)
            used_sources.add(chosen_pol["source_name"])
            save_state(state)
            published_count += 1
            time.sleep(15)

    # Slots 2 & 3: High-Engagement Stories from Different Outlets
    print("Publishing Slots 2 & 3 (Varied Outlets)...", flush=True)
    for c in candidates:
        if published_count >= 3:
            break
        if c["source_name"] in used_sources and len(used_sources) < len(candidates):
            continue
        if c["entry"].link in state["posted_urls"]:
            continue

        if publish_article(c["entry"], c["source_name"], c["img_url"], c["curated"]):
            state["posted_urls"].append(c["entry"].link)
            used_sources.add(c["source_name"])
            save_state(state)
            published_count += 1
            time.sleep(15)

    if published_count < 3:
        for c in candidates:
            if published_count >= 3:
                break
            if c["entry"].link in state["posted_urls"]:
                continue
            if publish_article(c["entry"], c["source_name"], c["img_url"], c["curated"]):
                state["posted_urls"].append(c["entry"].link)
                save_state(state)
                published_count += 1
                time.sleep(15)

    print(f"Cycle completed. Articles published: {published_count}", flush=True)

if __name__ == "__main__":
    main()

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
    {"name": "Jamuna TV", "url": "https://www.jamuna.tv/feed"},
    {"name": "BBC World", "url": "https://feeds.bbci.co.uk/news/world/rss.xml"},
    {"name": "Reuters", "url": "https://feedx.net/rss/reuters.xml"},
    {"name": "Al Jazeera", "url": "https://www.aljazeera.com/xml/rss/all.xml"}
]

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "bn,en-US;q=0.9,en;q=0.8",
    "Referer": "https://www.google.com/"
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
    return cleaned.strip()

def quick_bd_relevance_check(title, summary, is_international):
    combined = (title + " " + summary).lower()
    if is_international:
        return any(k in combined for k in ["bangladesh", "dhaka", "hasina", "yunus", "bengali", "rohingya"])
    return True

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

def query_llm_dual_engine(prompt):
    if groq_client and ACTIVE_GROQ_MODELS:
        for model in ACTIVE_GROQ_MODELS[:3]:
            try:
                res = groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=model,
                    temperature=0.3,
                    max_tokens=350,
                )
                text = res.choices[0].message.content.strip()
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
                if len(text) > 20:
                    return text
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
                    if res and res.text and len(res.text.strip()) > 20:
                        return res.text.strip()
                except Exception as gme:
                    err_msg = str(gme)
                    if "429" in err_msg and attempt == 0:
                        wait_sec = 21
                        m = re.search(r"retry in (\d+)", err_msg)
                        if m:
                            wait_sec = int(m.group(1)) + 1
                        time.sleep(wait_sec)
                    else:
                        break
    return None

def analyze_and_score_news(raw_title, raw_summary, source_name):
    clean_t = pre_clean_text(raw_title)
    clean_s = pre_clean_text(raw_summary) if raw_summary else clean_t

    prompt = f"""You are the Chief Editorial Strategist for 'Bongo Tribune', a premier digital news outlet in Bangladesh.
Evaluate this breaking news story for MAXIMUM AUDIENCE ENGAGEMENT & REACH:
Source: {source_name}
Title: {clean_t}
Summary: {clean_s}

CRITICAL EDITORIAL CRITERIA:
1. BANGLADESH FOCUS: Must be directly tied to Bangladesh national interest, people, or diaspora.
2. VIRAL / ENGAGEMENT SCORING (1 to 10):
   - Score 8-10: Massive public debate, breaking controversy, major political shock, high emotion, viral interest.
   - Score 5-7: Important standard news, regular updates, policy changes.
   - Score 1-4: Routine, boring, press releases, or minor municipal issues.
3. IS_POLITICS: YES if about national politics, interim government, elections, political party conflicts, high-profile arrests/trials.
4. COPYWRITING (Always 100% natural journalistic Bengali):
   - HEADLINE: High-impact, punchy, dramatic Bengali headline (max 10-14 words).
   - SUB_HEADLINE: Contextual sub-headline (or 'None').
   - SUMMARY: Exactly 2 crisp sentences summarizing the core development in fluent Bengali.

Format output with these exact labels:
RELEVANT: YES or NO
IS_POLITICS: YES or NO
ENGAGEMENT_SCORE: <integer 1 to 10>
HEADLINE: <bengali headline>
SUB_HEADLINE: <bengali sub-headline or None>
SUMMARY: <bengali summary>"""

    response_text = query_llm_dual_engine(prompt)
    if not response_text:
        return None

    try:
        clean_resp = re.sub(r"[*#_`]", "", response_text)
        is_intl = any(k in source_name.lower() for k in ["bbc world", "reuters", "ap news", "al jazeera"])
        if is_intl:
            rel_m = re.search(r"RELEVANT:\s*(YES|NO)", clean_resp, re.IGNORECASE)
            if rel_m and "NO" in rel_m.group(1).upper():
                return None

        pol_m = re.search(r"IS_POLITICS:\s*(YES|NO)", clean_resp, re.IGNORECASE)
        is_pol = bool(pol_m and "YES" in pol_m.group(1).upper())
        if not is_pol:
            is_pol = any(k in (clean_t + " " + clean_s).lower() for k in POLITICS_KEYWORDS)

        score_match = re.search(r"ENGAGEMENT_SCORE:\s*(\d+)", clean_resp, re.IGNORECASE)
        score = int(score_match.group(1)) if score_match else 6

        headline = clean_t
        sub_headline = ""
        summary = clean_s

        for line in clean_resp.split("\n"):
            line = line.strip()
            if re.match(r"^HEADLINE:\s*", line, re.IGNORECASE):
                hl = re.sub(r"^HEADLINE:\s*", "", line, flags=re.IGNORECASE).strip(' "')
                if len(hl) > 5:
                    headline = hl
            elif re.match(r"^SUB_HEADLINE:\s*", line, re.IGNORECASE):
                sub = re.sub(r"^SUB_HEADLINE:\s*", "", line, flags=re.IGNORECASE).strip(' "')
                if sub.lower() != "none" and len(sub) > 3:
                    sub_headline = sub
            elif re.match(r"^SUMMARY:\s*", line, re.IGNORECASE):
                sm = re.sub(r"^SUMMARY:\s*", "", line, flags=re.IGNORECASE).strip(' "')
                if len(sm) > 10:
                    summary = sm

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
    """Fallback search to retrieve an authentic high-resolution news photo."""
    try:
        search_term = urllib.parse.quote(f"{query} bangladesh news")
        url = f"https://html.duckduckgo.com/html/?q={search_term}"
        resp = requests.get(url, headers=BROWSER_HEADERS, timeout=8)
        img_candidates = re.findall(r'//external-content\.duckduckgo\.com/iu/\?u=(https?://[^&"\']+)', resp.text)
        for candidate in img_candidates:
            dec = urllib.parse.unquote(candidate)
            if dec.endswith(('.jpg', '.jpeg', '.png', '.webp')) and 'logo' not in dec.lower() and 'icon' not in dec.lower():
                return dec
    except Exception:
        pass
    return None

def extract_high_res_image(entry):
    headers = dict(BROWSER_HEADERS)
    headers["Referer"] = urllib.parse.urlsplit(entry.link).scheme + "://" + urllib.parse.urlsplit(entry.link).netloc
    
    # 1. Inspect RSS Enclosures / Media Content first
    if 'media_content' in entry and len(entry.media_content) > 0:
        url = entry.media_content[0].get('url')
        if url and not url.endswith(('.svg', '.gif')):
            return clean_and_maximize_image_url(url)

    if 'enclosures' in entry and len(entry.enclosures) > 0:
        url = entry.enclosures[0].get('href')
        if url and not url.endswith(('.svg', '.gif')):
            return clean_and_maximize_image_url(url)

    # 2. Scrape Article HTML OpenGraph & JSON-LD
    try:
        resp = requests.get(entry.link, timeout=9, headers=headers)
        if resp.status_code == 200:
            html = resp.text
            json_ld_matches = re.findall(r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', html, re.DOTALL | re.IGNORECASE)
            for jld in json_ld_matches:
                try:
                    data = json.loads(jld.strip())
                    if isinstance(data, dict):
                        img = data.get("image")
                        if isinstance(img, str) and img.startswith("http"):
                            return clean_and_maximize_image_url(img)
                        elif isinstance(img, dict) and "url" in img:
                            return clean_and_maximize_image_url(img["url"])
                        elif isinstance(img, list) and len(img) > 0 and isinstance(img[0], str):
                            return clean_and_maximize_image_url(img[0])
                except Exception:
                    pass

            patterns = [
                r'<meta[^>]+property=["\']og:image:secure_url["\'][^>]+content=["\'](https?://[^"\']+)["\']',
                r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\'](https?://[^"\']+)["\']',
                r'<meta[^>]+content=["\'](https?://[^"\']+)["\'][^>]+property=["\']og:image["\']',
                r'<meta[^>]+name=["\']twitter:image["\'][^>]+content=["\'](https?://[^"\']+)["\']',
                r'<link[^>]+rel=["\']image_src["\'][^>]+href=["\'](https?://[^"\']+)["\']'
            ]
            for pat in patterns:
                m = re.search(pat, html, re.IGNORECASE)
                if m:
                    candidate = m.group(1)
                    if not candidate.endswith(('.svg', '.gif', '.ico')) and 'avatar' not in candidate.lower() and 'logo' not in candidate.lower():
                        return clean_and_maximize_image_url(candidate)
    except Exception:
        pass

    # 3. Fallback to Related Image Search
    clean_title = pre_clean_text(entry.title)
    return search_related_news_image(clean_title)

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

def download_image_robust(url):
    """Reliably downloads high-res images circumventing newspaper anti-bot protection."""
    if not url:
        return None
    session = requests.Session()
    headers = dict(BROWSER_HEADERS)
    parsed = urllib.parse.urlsplit(url)
    headers["Host"] = parsed.netloc
    headers["Referer"] = f"{parsed.scheme}://{parsed.netloc}/"
    
    try:
        resp = session.get(url, timeout=12, headers=headers)
        if resp.status_code == 200 and len(resp.content) > 4000:
            return Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception:
        pass
    return None

def build_bongo_card(image_url, headline, sub_headline, summary, source_name, is_square=False):
    width = 1080
    height = 1080 if is_square else 1350
    top_h = int(height * 0.62)
    card = Image.new("RGB", (width, height), color="#4c0000")

    # 1. PHOTO RENDERING (Top 62% with Gaussian Blur Ambient Filler)
    raw_img = download_image_robust(image_url)
    if not raw_img:
        # Fallback to search query
        query_fallback = search_related_news_image(headline)
        raw_img = download_image_robust(query_fallback)

    if raw_img:
        try:
            bg_blur = raw_img.resize((width, top_h), Image.Resampling.BILINEAR)
            bg_blur = bg_blur.filter(ImageFilter.GaussianBlur(radius=30))

            scale = max(width / raw_img.width, top_h / raw_img.height)
            new_w = max(1, int(raw_img.width * scale))
            new_h = max(1, int(raw_img.height * scale))
            fit_img = raw_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            paste_x = (width - new_w) // 2
            paste_y = (top_h - new_h) // 2
            bg_blur.paste(fit_img, (paste_x, paste_y))
            card.paste(bg_blur, (0, 0))
        except Exception as e:
            print(f"Image processing note: {e}", flush=True)

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

    # 3. EXPANDED CHAT BUBBLE (Matches BT NEWS CARD mockup)
    bubble_w = 930
    bubble_h = 560 if is_square else 650
    bubble_x = (width - bubble_w) // 2
    
    # Placed right above the bottom footer labels
    footer_margin = 85
    bubble_y = height - bubble_h - footer_margin

    bubble_img = Image.new("RGBA", (bubble_w, bubble_h), (0, 0, 0, 0))
    b_draw = ImageDraw.Draw(bubble_img)
    tail_h = 45
    
    # White Bubble Box
    b_draw.rounded_rectangle([(0, 0), (bubble_w, bubble_h - tail_h)], radius=26, fill=(255, 255, 255, 255))
    
    # Pointing speech tail placed right above the source
    tail = [(bubble_w - 150, bubble_h - tail_h), (bubble_w - 55, bubble_h), (bubble_w - 55, bubble_h - tail_h)]
    b_draw.polygon(tail, fill=(255, 255, 255, 255))

    # Watermark Seal (#4c0000 with 20% opacity)
    watermark_path = get_asset("watermark")
    if watermark_path:
        try:
            wm = Image.open(watermark_path).convert("RGBA")
            wm_size = int(bubble_h * 0.68)
            wm = wm.resize((wm_size, wm_size), Image.Resampling.LANCZOS)
            r, g, b, a = wm.split()
            new_alpha = a.point(lambda p: int(p * 0.20))
            tinted_wm = Image.new("RGBA", (wm_size, wm_size), color=(76, 0, 0, 0))
            tinted_wm.putalpha(new_alpha)
            wm_x = (bubble_w - wm_size) // 2
            wm_y = (bubble_h - tail_h - wm_size) // 2
            bubble_img.paste(tinted_wm, (wm_x, wm_y), mask=tinted_wm)
        except Exception:
            pass

    font_hl = get_font(46 if not is_square else 38)
    font_sub = get_font(32 if not is_square else 27)
    font_sum = get_font(28 if not is_square else 24)

    text_pad_x = 45
    text_y = 38
    inner_w = bubble_w - (text_pad_x * 2)

    # Headline: bold maroon #4c0000
    hl_lines = wrap_text(headline, font_hl, inner_w, b_draw)
    for line in hl_lines[:3]:
        bbox = b_draw.textbbox((0, 0), line, font=font_hl)
        line_w = bbox[2] - bbox[0]
        b_draw.text((text_pad_x + (inner_w - line_w) // 2, text_y), line, fill="#4c0000", font=font_hl)
        text_y += (bbox[3] - bbox[0]) + 12

    # Sub-headline: bold maroon #4c0000
    if sub_headline:
        text_y += 6
        sub_lines = wrap_text(sub_headline, font_sub, inner_w, b_draw)
        for line in sub_lines[:2]:
            bbox = b_draw.textbbox((0, 0), line, font=font_sub)
            line_w = bbox[2] - bbox[0]
            b_draw.text((text_pad_x + (inner_w - line_w) // 2, text_y), line, fill="#4c0000", font=font_sub)
            text_y += (bbox[3] - bbox[0]) + 10

    # Summary: bold pure black #000000
    text_y += 16
    sum_lines = wrap_text(summary, font_sum, inner_w, b_draw)
    for line in sum_lines[:4]:
        bbox = b_draw.textbbox((0, 0), line, font=font_sum)
        line_w = bbox[2] - bbox[0]
        b_draw.text((text_pad_x + (inner_w - line_w) // 2, text_y), line, fill="#000000", font=font_sum)
        text_y += (bbox[3] - bbox[0]) + 8

    card.paste(bubble_img, (bubble_x, bubble_y), mask=bubble_img)

    # 4. FOOTER: DATE & SOURCE ALIGNMENT
    draw = ImageDraw.Draw(card)
    font_footer = get_font(26)
    date_str = datetime.utcnow().strftime("%d %B").upper()
    footer_y = height - 60
    
    # Left aligned date
    draw.text((60, footer_y), date_str, fill="#ffffff", font=font_footer)

    # Right aligned source
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
    for _ in range(8):
        time.sleep(5)
        res = requests.get(status_url).json()
        if res.get("status_code") == "FINISHED":
            return True
        if res.get("status_code") == "ERROR":
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
    
    print(f"Publishing article (Score {curated.get('score')}): {headline}", flush=True)
    
    fb_card_path = build_bongo_card(img_url, headline, sub_headline, summary, source_name, is_square=False)
    if not fb_card_path:
        return False
        
    ig_card_path = build_bongo_card(img_url, headline, sub_headline, summary, source_name, is_square=True)
    
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
                ig_caption = f"{headline}\n\n{summary}\n\nসূত্র: {source_name}\n\n#bongotribune #banglanews #bangladesh #news"
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
            print(f"Instagram posting error: {err}", flush=True)

    return True

def scan_feeds_smart(state):
    print(f"Smart-scanning {len(ALL_FEEDS)} media feeds with Dual-AI engine...", flush=True)
    qualifying_candidates = []
    prefiltered_entries = []

    # Interleaved round-robin scan across all media channels
    feed_entries_map = {}
    for feed in ALL_FEEDS:
        try:
            parsed = feedparser.parse(feed["url"])
            is_intl = any(k in feed["name"].lower() for k in ["bbc world", "reuters", "ap news", "al jazeera"])
            valid_for_this_feed = []
            
            for entry in parsed.entries[:6]:
                if entry.link in state["posted_urls"]:
                    continue
                clean_t = pre_clean_text(entry.title)
                if len(clean_t.split()) < 3:
                    continue
                
                raw_summary = entry.get("summary", "")
                if not quick_bd_relevance_check(clean_t, raw_summary, is_intl):
                    continue
                
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

    # Round-robin selection across unique channels to ensure variance
    max_depth = max([len(v) for v in feed_entries_map.values()]) if feed_entries_map else 0
    for depth in range(min(max_depth, 2)):
        for source_name, entries in feed_entries_map.items():
            if depth < len(entries):
                prefiltered_entries.append(entries[depth])

    print(f"Pre-filter gathered {len(prefiltered_entries)} varied candidate stories across different newspapers.", flush=True)

    # Evaluate candidates with Dual-AI
    seen_sources = set()
    evaluation_queue = []
    
    # Prioritize 1 story per source first
    for e in prefiltered_entries:
        if e["source"] not in seen_sources:
            seen_sources.add(e["source"])
            evaluation_queue.append(e)
        if len(evaluation_queue) >= 12:
            break

    for item in evaluation_queue:
        entry = item["entry"]
        img_url = extract_high_res_image(entry)
        curated = analyze_and_score_news(entry.title, entry.get("summary", ""), item["source"])
        if curated:
            qualifying_candidates.append({
                "entry": entry,
                "source_name": item["source"],
                "img_url": img_url,
                "curated": curated,
                "is_politics": curated["is_politics"],
                "score": curated["score"]
            })
            print(f"Evaluated: [{item['source']}] {curated['headline']} | Score: {curated['score']} | Politics: {curated['is_politics']}", flush=True)
        time.sleep(2)

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
            time.sleep(20)

    # Slots 2 & 3: Highest Engagement News with Source Diversity
    print("Publishing Slots 2 & 3 (Varied Media Outlets)...", flush=True)
    for c in candidates:
        if published_count >= 3:
            break
        # Ensure variance: Avoid duplicate sources in the same run
        if c["source_name"] in used_sources and len(used_sources) < len(candidates):
            continue
        if c["entry"].link in state["posted_urls"]:
            continue

        if publish_article(c["entry"], c["source_name"], c["img_url"], c["curated"]):
            state["posted_urls"].append(c["entry"].link)
            used_sources.add(c["source_name"])
            save_state(state)
            published_count += 1
            time.sleep(20)

    # If diverse sources ran out, fill remaining slots with next highest scores
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
                time.sleep(20)

    print(f"Cycle completed. Articles published: {published_count}", flush=True)

if __name__ == "__main__":
    main()

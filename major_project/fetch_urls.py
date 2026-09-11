import urllib.request
import csv
import random
import os

OUTPUT_FILE = r"d:\major_project\data\urls.csv"

def generate_dataset():
    urls = []
    
    print("Fetching latest Phishing URLs from OpenPhish feed...")
    try:
        req = urllib.request.Request("https://openphish.com/feed.txt", headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            phish_urls = response.read().decode('utf-8').strip().split('\n')
            for u in phish_urls:
                if u.strip():
                    urls.append((u.strip(), 1))
            print(f"Retrieved {len(phish_urls)} real phishing URLs.")
    except Exception as e:
        print(f"Failed to fetch phishing list: {e}")

    # Ensure we have at least 2500 phishing URLs
    if sum(1 for _, label in urls if label == 1) < 2500:
        print("Padding with synthetic phishing data to reach 2500...")
        keywords = ["login", "verify", "update", "secure", "account", "banking", "service", "helpdesk", "auth"]
        current_phish = sum(1 for _, label in urls if label == 1)
        needed = 2500 - current_phish
        for i in range(needed):
            kw = random.choice(keywords)
            urls.append((f"http://{kw}-info-{random.randint(1,99999)}.net/authenticate?id={i}", 1))

    print("Generating legitimate URLs...")
    domains = [
        "google.com", "youtube.com", "facebook.com", "wikipedia.org", "yahoo.com", 
        "twitter.com", "amazon.com", "instagram.com", "linkedin.com", "reddit.com", 
        "netflix.com", "ebay.com", "microsoft.com", "apple.com", "twitch.tv", 
        "tumblr.com", "imdb.com", "github.com", "pinterest.com", "stackoverflow.com",
        "medium.com", "spotify.com", "dropbox.com", "zoom.us", "salesforce.com"
    ]
    
    # Generate random safe-looking urls to balance the dataset
    target_legit = max(2500, 5000 - len(urls)) # If we got 3000 phish, we need at least 2000 legit, we'll just aim for total 5000.
    count = 0
    while count < target_legit:
        domain = random.choice(domains)
        paths = ["", "home", "about", "contact", "support", "dashboard", "profile", "settings"]
        path = random.choice(paths)
        if random.random() > 0.7:
             path += f"?session_id={random.randint(1000, 999999)}"
        urls.append((f"https://www.{domain}/{path}", 0))
        count += 1

    # Shuffle to mix phishing and legitimate
    random.shuffle(urls)
    
    # Slice to exactly 5000
    final_urls = urls[:5000]

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['url', 'label'])
        writer.writerows(final_urls)
        
    phish_count = sum(1 for _, label in final_urls if label == 1)
    legit_count = sum(1 for _, label in final_urls if label == 0)
    print(f"Successfully wrote {len(final_urls)} URLs to {OUTPUT_FILE}")
    print(f"Phishing: {phish_count}, Legitimate: {legit_count}")

if __name__ == '__main__':
    generate_dataset()

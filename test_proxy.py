import requests

proxies_list = [
    "socks4://185.32.4.110:4153",
    "socks4://8.220.204.215:4002",
    "socks4://108.175.23.49:13135",
    "socks4://161.49.176.173:1338",
    "socks4://175.144.198.226:31694",
    "socks4://47.254.36.213:999",
]

for proxy in proxies_list:
    proxies = {
        "http": proxy,
        "https": proxy,
    }
    try:
        print(f"Testing proxy {proxy}...")
        response = requests.get("https://httpbin.org/ip", proxies=proxies, timeout=5)
        print("Success:", response.json())
    except Exception as e:
        print("Failed:", e)

import os
from flask import Flask

app = Flask(__name__)

# আপনার টেলিগ্রাম বট বা অন্যান্য কোড এখানে থাকবে...

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)

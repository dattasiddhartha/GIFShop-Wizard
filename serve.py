from waitress import serve
#import app
#serve(app.app, host='0.0.0.0', port=8000)

import newapp
serve(newapp.app, host='0.0.0.0', port=8000)
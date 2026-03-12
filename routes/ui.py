from flask import Blueprint, render_template

ui_bp = Blueprint('ui', __name__, 
                  template_folder='/app/templates',
                  static_folder='/app/static')
@ui_bp.route("/")
def dashboard():
    return render_template("index.html")
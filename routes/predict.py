from flask import Blueprint, request, jsonify
from services.inference_service import get_sensor_values, run_analysis

predict_bp = Blueprint("predict", __name__)

@predict_bp.route("/get_scenario_sensors", methods=["POST"])
def get_scenario_sensors():
    """
    Get sensors for a specific scenario (normal, warning, critical)
    """
    data = request.json
    scenario = data.get("scenario", "normal")
    
    engine_id, cycle, sensors = get_sensor_values(scenario)
    
    if sensors is None:
        return jsonify({"error": "No data found for this scenario"}), 404
    
    return jsonify({
        "engine_id": engine_id,
        "cycle": cycle,
        "sensors": sensors
    })

@predict_bp.route("/analyse", methods=["POST"])
def analyse():
    """
    Run analysis on current engine/cycle
    """
    data = request.json
    engine_id = data.get("engine_id")
    cycle = data.get("cycle")
    
    if not engine_id or not cycle:
        return jsonify({"error": "Missing engine_id or cycle"}), 400
    
    result = run_analysis(engine_id, cycle)
    
    if "error" in result:
        return jsonify(result), 404
    
    return jsonify(result)
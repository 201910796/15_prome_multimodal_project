import PropTypes from "prop-types";
import "./Recommendation.css";

const Recommendation = ({ className = "" }) => {
  return (
    <div className={`recommendation ${className}`}>
      <div className="recommendation-child" />
      <div className="button1">
        <img className="icon2" alt="" src="/icon-1@2x.png" />
        <div className="ai-wrapper">
          <div className="ai">AI에게 음악 추천받기</div>
        </div>
      </div>
      <div className="music-examples">
        <div className="music1">
          <div className="music-icons">
            <img className="icon3" alt="" src="/icon-2.svg" />
          </div>
          <div className="bruno-mars-">로제, Bruno Mars - APT.</div>
        </div>
        <div className="music2">
          <div className="music-icons">
            <img className="icon3" alt="" src="/icon-2.svg" />
          </div>
          <div className="div21">혜원 - 마루는 강쥐</div>
        </div>
      </div>
    </div>
  );
};

Recommendation.propTypes = {
  className: PropTypes.string,
};

export default Recommendation;

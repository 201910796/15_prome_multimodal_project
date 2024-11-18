import PropTypes from "prop-types";
import "./AddMusic.css";

const AddMusic = ({ className = "" }) => {
  return (
    <div className={`add-music1 ${className}`}>
      <div className="icon-parent">
        <img className="icon1" alt="" src="/icon@2x.png" />
        <div className="wrapper3">
          <div className="div20">음악 추가</div>
        </div>
      </div>
      <div className="add-music-detail">
        <div className="button">
          <div className="add-music-detail-background" />
          <img className="detail-icon" alt="" src="/detail@2x.png" />
        </div>
      </div>
    </div>
  );
};

AddMusic.propTypes = {
  className: PropTypes.string,
};

export default AddMusic;

import PropTypes from "prop-types";
import "./FrameComponent4.css";

const FrameComponent4 = ({ className = "" }) => {
  return (
    <div className={`frame-parent1 ${className}`}>
      <div className="button-wrapper">
        <div className="button2">
          <div className="share-background" />
          <div className="div22">공유</div>
        </div>
      </div>
      <div className="bars-home-indicator4">
        <footer className="background6" />
        <div className="line4" />
      </div>
    </div>
  );
};

FrameComponent4.propTypes = {
  className: PropTypes.string,
};

export default FrameComponent4;

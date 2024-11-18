import PropTypes from "prop-types";
import "./FrameComponent3.css";

const FrameComponent3 = ({ className = "" }) => {
  return (
    <div className={`bars-status-bar-iphone-x-parent ${className}`}>
      <header className="bars-status-bar-iphone-x1">
        <div className="background5" />
        <div className="div19">9:41</div>
        <div className="signal-icons-wrapper">
          <div className="signal-icons">
            <img
              className="mobile-signal-icon1"
              loading="lazy"
              alt=""
              src="/mobile-signal.svg"
            />
            <img className="wifi-icon1" loading="lazy" alt="" src="/wifi.svg" />
            <img
              className="battery-icon1"
              loading="lazy"
              alt=""
              src="/battery@2x.png"
            />
          </div>
        </div>
      </header>
      <div className="frame-container">
        <div className="frame-group">
          <div className="back-wrapper">
            <img className="back-icon" loading="lazy" alt="" src="/back.svg" />
          </div>
          <a className="a">새 게시물</a>
        </div>
      </div>
    </div>
  );
};

FrameComponent3.propTypes = {
  className: PropTypes.string,
};

export default FrameComponent3;

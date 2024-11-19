import PropTypes from "prop-types";
import "./FrameComponent5.css";

const FrameComponent5 = ({ className = "" }) => {
  return (
    <div className={`bars-status-bar-iphone-x-group ${className}`}>
      <header className="bars-status-bar-iphone-x3">
        <div className="background8" />
        <img
          className="mobile-signal-icon3"
          loading="lazy"
          alt=""
          src="/mobile-signal.svg"
        />
        <img className="wifi-icon3" loading="lazy" alt="" src="/wifi.svg" />
        <img
          className="battery-icon3"
          loading="lazy"
          alt=""
          src="/battery@2x.png"
        />
      </header>
      <div className="content">
        <div className="navigation">
          <div className="title-area">
            <div className="page-title">9:41</div>
          </div>
          <div className="header">
            <div className="back-button-area">
              <img
                className="back-icon1"
                loading="lazy"
                alt=""
                src="/back.svg"
              />
            </div>
            <a className="a1">게시물</a>
          </div>
        </div>
      </div>
    </div>
  );
};

FrameComponent5.propTypes = {
  className: PropTypes.string,
};

export default FrameComponent5;

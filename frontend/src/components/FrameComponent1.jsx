import PropTypes from "prop-types";
import "./FrameComponent1.css";

const FrameComponent1 = ({ className = "" }) => {
  return (
    <div className={`next-group ${className}`}>
      <a className="next1">다음</a>
      <img
        className="select-multiple-icon1"
        loading="lazy"
        alt=""
        src="/select-multiple1@2x.png"
      />
      <header className="bars-status-bar-iphone-x2">
        <div className="background7" />
        <img className="wifi-icon2" loading="lazy" alt="" src="/wifi.svg" />
        <img
          className="battery-icon2"
          loading="lazy"
          alt=""
          src="/battery@2x.png"
        />
      </header>
      <div className="recents-parent">
        <a className="recents2">새 게시물</a>
        <div className="mobile-signal-parent">
          <img
            className="mobile-signal-icon2"
            loading="lazy"
            alt=""
            src="/mobile-signal.svg"
          />
          <div className="frame-parent2">
            <div className="placeholder-wrapper">
              <div className="placeholder">9:41</div>
            </div>
            <img
              className="close-icon1"
              loading="lazy"
              alt=""
              src="/close-icon.svg"
            />
            <div className="recents-group">
              <div className="recents3">최근</div>
              <div className="other-albums-wrapper">
                <img
                  className="other-albums-icon1"
                  alt=""
                  src="/other-albums.svg"
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

FrameComponent1.propTypes = {
  className: PropTypes.string,
};

export default FrameComponent1;

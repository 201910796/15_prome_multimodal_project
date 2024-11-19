import PropTypes from "prop-types";
import "./FrameComponent2.css";

const FrameComponent2 = ({ className = "" }) => {
  return (
    <div className={`next-parent ${className}`}>
      <a className="next">다음</a>
      <header className="bars-status-bar-iphone-x">
        <div className="background4" />
        <img
          className="battery-icon"
          loading="lazy"
          alt=""
          src="/battery@2x.png"
        />
      </header>
      <div className="top-navigation">
        <a className="recents">새 게시물</a>
        <div className="options">
          <img
            className="select-multiple-icon"
            loading="lazy"
            alt=""
            src="/select-multiple@2x.png"
          />
          <div className="options-list">
            <div className="option-item-one">
              <div className="option-name">
                <div className="option-description">9:41</div>
              </div>
              <img
                className="close-icon"
                loading="lazy"
                alt=""
                src="/close-icon.svg"
              />
              <div className="option-item-two">
                <div className="recents1">최근</div>
                <div className="other-options">
                  <img
                    className="other-albums-icon"
                    alt=""
                    src="/other-albums.svg"
                  />
                </div>
              </div>
            </div>
            <img className="wifi-icon" loading="lazy" alt="" src="/wifi.svg" />
            <img
              className="mobile-signal-icon"
              loading="lazy"
              alt=""
              src="/mobile-signal.svg"
            />
          </div>
        </div>
      </div>
    </div>
  );
};

FrameComponent2.propTypes = {
  className: PropTypes.string,
};

export default FrameComponent2;

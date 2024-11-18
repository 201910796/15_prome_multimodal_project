import PropTypes from "prop-types";
import "./TopBar.css";

const TopBar = ({ className = "" }) => {
  return (
    <div className={`top-bar ${className}`}>
      <div className="top-bar-background" />
      <header className="bars-status-bar-iphone-x4">
        <div className="background10" />
        <div className="carrier">9:41</div>
        <div className="status-bar-right">
          <div className="network-status">
            <img
              className="mobile-signal-icon4"
              loading="lazy"
              alt=""
              src="/mobile-signal.svg"
            />
            <img className="wifi-icon4" loading="lazy" alt="" src="/wifi.svg" />
            <img
              className="battery-icon4"
              loading="lazy"
              alt=""
              src="/battery@2x.png"
            />
          </div>
        </div>
      </header>
      <div className="user-info">
        <div className="jacob-w-parent">
          <a className="jacob-w">promi1004</a>
          <div className="user-actions">
            <img className="add-icon" loading="lazy" alt="" src="/add@2x.png" />
            <div className="menu-wrapper">
              <img
                className="menu-icon"
                loading="lazy"
                alt=""
                src="/menu@2x.png"
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

TopBar.propTypes = {
  className: PropTypes.string,
};

export default TopBar;

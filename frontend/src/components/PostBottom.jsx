import PropTypes from "prop-types";
import "./PostBottom.css";
import React, {useState} from 'react';

const PostBottom = ({ className = "" }) => {

  const [isLiked, setIsLiked] = useState(false);

  const handleClick = () => {
    setIsLiked(!isLiked);
  };

  return (
    <section className={`post-bottom ${className}`}>
      <div className="post-bottom1">
        <div className="rectangle2" />
        <div className="interaction-area">
          <div className="like-save-area">
            <div className="action-buttons">
              <div className="like-button">
                <img
                  className="image-icon2"
                  loading="lazy"
                  alt=""
                  src="/image-1@2x.png"
                />
                {/* <img
                  className="like-icon"
                  loading="lazy"
                  alt=""
                  src="/like@2x.png"
                /> */}
               <img
                  className={`like-icon ${isLiked ? 'active' : ''}`}
                  loading="lazy"
                  alt=""
                  src="/like@2x.png"
                  onClick={handleClick}
                />
              </div>
              <div className="comment-pagination-parent">
                <div className="comment-pagination">
                  <div className="comment-pagination-actions">
                    <div className="pagination-buttons">
                      <img
                        className="comment-icon"
                        loading="lazy"
                        alt=""
                        src="/comment@2x.png"
                      />
                      <img
                        className="messanger-icon"
                        loading="lazy"
                        alt=""
                        src="/messanger@2x.png"
                      />
                    </div>
                    <div className="pagination-area">
                      <img
                        className="pagination-icon"
                        loading="lazy"
                        alt=""
                        src="/pagination.svg"
                      />
                    </div>
                  </div>
                </div>
                <div className="liked-by-craig-love-container">
                  <span className="prometheusai">prometheus.ai</span>
                  <span>{`님 외 `}</span>
                  <span className="prometheusai">여러 명</span>
                  <span>이 좋아합니다</span>
                </div>
              </div>
            </div>
            <img
              className="save-icon"
              loading="lazy"
              alt=""
              src="/save@2x.png"
            />
          </div>
          <div className="caption">
            <div className="joshua-l-the-game-container">
              <span className="prometheusai">promi123</span>
              <span>
                {" "}
                프로미가 좋아하는 랜덤 게임 게임 스타트 아파트 아파트 아파트
                아파트 1, 2, 3
              </span>
            </div>
          </div>
        </div>
        <div className="timestamp">
          <div className="september-19">10월 1일</div>
        </div>
      </div>
      <img
        className="tab-bar-icon1"
        loading="lazy"
        alt=""
        src="/tab-bar@2x.png"
      />
      <div className="bars-home-indicator5">
        <footer className="background9" />
        <div className="line5" />
      </div>
    </section>
  );
};

PostBottom.propTypes = {
  className: PropTypes.string,
};

export default PostBottom;

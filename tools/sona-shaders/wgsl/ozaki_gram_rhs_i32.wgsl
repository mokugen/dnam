struct type_6_ {
    member: array<f32>,
}

struct type_12_ {
    member: u32,
    member_1_: u32,
}

struct type_14_ {
    member: f32,
    member_1_: f32,
}

struct type_15_ {
    member: u32,
    member_1_: u32,
    member_2_: u32,
    member_3_: u32,
    member_4_: u32,
    member_5_: u32,
    member_6_: u32,
    member_7_: u32,
}

struct type_16_ {
    member: type_15_,
}

struct type_19_ {
    member: array<vec2<f32>>,
}

struct type_22_ {
    member: array<i32>,
}

struct type_28_ {
    member: u32,
    member_1_: u32,
    member_2_: u32,
    member_3_: u32,
    member_4_: bool,
    member_5_: bool,
}

var<private> wg_id_1_: vec3<u32>;
var<private> lid_1_: u32;
@group(0) @binding(0) 
var<uniform> uniforms: type_16_;
@group(0) @binding(1) 
var<storage> features: type_6_;
@group(0) @binding(2) 
var<storage> targets: type_6_;
@group(0) @binding(3) 
var<storage> col_scales: type_19_;
@group(0) @binding(4) 
var<storage, read_write> ozaki_buckets: type_22_;
var<workgroup> reduce_i32_: array<i32, 1280>;

fn function_() {
    var local: array<i32, 5>;
    var local_1_: array<i32, 5>;
    var phi_239_: u32;
    var phi_242_: u32;
    var phi_260_: u32;
    var phi_240_: u32;
    var phi_243_: u32;
    var local_2_: u32;
    var local_3_: u32;
    var local_4_: u32;
    var phi_266_: type_28_;
    var phi_267_: type_28_;
    var phi_268_: bool;
    var phi_273_: type_28_;
    var phi_192_: u32;
    var phi_195_: u32;
    var phi_217_: u32;
    var phi_193_: u32;
    var phi_196_: u32;
    var local_5_: u32;
    var local_6_: u32;
    var local_7_: u32;
    var local_8_: u32;
    var phi_274_: type_28_;
    var phi_275_: type_28_;
    var phi_276_: bool;
    var phi_280_: type_28_;
    var phi_281_: type_28_;
    var phi_1259_: u32;
    var phi_1262_: type_12_;
    var phi_1264_: type_12_;
    var phi_1289_: u32;
    var phi_1308_: u32;
    var phi_1327_: u32;
    var phi_1346_: u32;
    var phi_1365_: u32;
    var phi_1504_: type_12_;
    var phi_1505_: type_12_;
    var phi_1260_: u32;
    var phi_1263_: type_12_;
    var phi_1265_: type_12_;
    var phi_1928_: bool;
    var local_9_: type_12_;
    var local_10_: type_12_;
    var local_11_: type_12_;
    var local_12_: type_12_;
    var phi_1544_: u32;
    var phi_1545_: u32;
    var phi_1962_: bool;
    var phi_1717_: u32;
    var phi_1737_: u32;
    var phi_1757_: u32;
    var phi_1777_: u32;
    var phi_1798_: type_12_;
    var phi_1799_: type_12_;
    var phi_1811_: u32;
    var phi_1831_: u32;
    var phi_1851_: u32;
    var phi_1871_: u32;
    var phi_300_: u32;
    var phi_322_: u32;
    var phi_364_: bool;
    var phi_365_: bool;
    var phi_366_: f32;
    var phi_368_: bool;
    var phi_369_: f32;
    var phi_370_: bool;
    var phi_371_: bool;
    var phi_378_: f32;
    var phi_1908_: bool;
    var phi_415_: u32;
    var phi_422_: type_14_;
    var phi_423_: type_14_;
    var phi_424_: bool;
    var phi_425_: type_14_;
    var phi_426_: bool;
    var phi_431_: type_14_;
    var phi_433_: f32;
    var phi_465_: bool;
    var phi_466_: bool;
    var phi_467_: f32;
    var phi_469_: bool;
    var phi_470_: f32;
    var phi_471_: bool;
    var phi_472_: bool;
    var phi_479_: f32;
    var phi_1923_: bool;
    var phi_516_: u32;
    var phi_523_: type_14_;
    var phi_524_: type_14_;
    var phi_525_: bool;
    var phi_526_: type_14_;
    var phi_527_: bool;
    var phi_532_: type_14_;
    var phi_534_: f32;
    var phi_541_: bool;
    var phi_544_: type_12_;
    var phi_547_: array<i32, 6>;
    var phi_545_: type_12_;
    var phi_570_: type_12_;
    var phi_605_: u32;
    var phi_651_: u32;
    var phi_630_: u32;
    var phi_659_: f32;
    var phi_734_: i32;
    var phi_764_: i32;
    var phi_794_: i32;
    var phi_797_: array<i32, 3>;
    var phi_798_: array<i32, 3>;
    var phi_800_: array<i32, 3>;
    var phi_801_: bool;
    var phi_806_: array<i32, 3>;
    var phi_881_: i32;
    var phi_911_: i32;
    var phi_941_: i32;
    var phi_944_: array<i32, 3>;
    var phi_945_: array<i32, 3>;
    var phi_947_: array<i32, 3>;
    var phi_948_: bool;
    var phi_953_: array<i32, 3>;
    var phi_1000_: array<i32, 6>;
    var phi_1001_: array<i32, 6>;
    var phi_1002_: array<i32, 6>;
    var phi_548_: array<i32, 6>;
    var phi_1966_: bool;
    var local_13_: array<i32, 6>;
    var local_14_: array<i32, 6>;
    var local_15_: array<i32, 6>;
    var local_16_: array<i32, 6>;
    var local_17_: array<i32, 6>;
    var phi_1017_: type_12_;
    var phi_1018_: type_12_;
    var phi_1041_: type_12_;
    var phi_2049_: bool;
    var phi_2059_: bool;
    var phi_1076_: u32;
    var phi_1091_: type_12_;
    var phi_1092_: type_12_;
    var phi_1115_: type_12_;
    var phi_2055_: bool;
    var phi_2062_: bool;
    var phi_2061_: bool;
    var phi_1077_: u32;
    var phi_2060_: bool;
    var phi_1180_: type_12_;
    var phi_1181_: type_12_;
    var phi_1204_: type_12_;
    var phi_1230_: u32;
    var phi_2063_: bool;
    var phi_1248_: u32;

    switch bitcast<i32>(0u) {
        default: {
            let _e72_ = wg_id_1_;
            let _e73_ = lid_1_;
            let _e75_ = arrayLength((&features.member));
            let _e77_ = arrayLength((&targets.member));
            let _e79_ = arrayLength((&col_scales.member));
            let _e81_ = arrayLength((&ozaki_buckets.member));
            let _e82_ = (_e73_ & 255u);
            let _e85_ = uniforms.member.member_5_;
            let _e91_ = ((_e72_.y * select(256u, _e85_, (_e85_ > 0u))) + _e72_.x);
            let _e94_ = uniforms.member.member;
            let _e97_ = uniforms.member.member_1_;
            let _e100_ = uniforms.member.member_2_;
            let _e103_ = uniforms.member.member_3_;
            let _e105_ = select(_e103_, 1024u, (1024u < _e103_));
            let _e108_ = uniforms.member.member_4_;
            let _e110_ = select(1u, _e108_, (1u < _e108_));
            let _e114_ = ((_e97_ * (_e97_ + 1u)) >> bitcast<u32>(1i));
            let _e115_ = (_e97_ * _e100_);
            let _e117_ = ((_e114_ + _e115_) * 4u);
            if (_e97_ == 0u) {
                phi_281_ = type_28_(0u, 0u, 0u, 0u, false, true);
            } else {
                if (_e91_ < _e114_) {
                    phi_192_ = (_e97_ - 1u);
                    phi_195_ = 0u;
                    loop {
                        let _e165_ = phi_192_;
                        let _e167_ = phi_195_;
                        local_5_ = _e167_;
                        local_6_ = _e167_;
                        local_7_ = _e167_;
                        local_8_ = _e167_;
                        let _e168_ = (_e167_ < _e165_);
                        if _e168_ {
                            let _e173_ = (_e167_ + (((_e165_ - _e167_) + 1u) >> bitcast<u32>(1i)));
                            let _e174_ = (_e97_ - _e173_);
                            let _e180_ = ((_e114_ - ((_e174_ * (_e174_ + 1u)) >> bitcast<u32>(1i))) <= _e91_);
                            if _e180_ {
                                phi_217_ = _e165_;
                            } else {
                                phi_217_ = (_e173_ - 1u);
                            }
                            let _e183_ = phi_217_;
                            phi_193_ = _e183_;
                            phi_196_ = select(_e167_, _e173_, _e180_);
                        } else {
                            phi_193_ = u32();
                            phi_196_ = u32();
                        }
                        let _e186_ = phi_193_;
                        let _e188_ = phi_196_;
                        phi_192_ = _e186_;
                        phi_195_ = _e188_;
                        if !(_e168_) {
                            break;
                        }
                    }
                    let _e191_ = local_5_;
                    let _e192_ = (_e97_ - _e191_);
                    let _e200_ = local_6_;
                    let _e201_ = (_e200_ + (_e91_ - (_e114_ - ((_e192_ * (_e192_ + 1u)) >> bitcast<u32>(1i)))));
                    let _e203_ = local_7_;
                    let _e205_ = local_8_;
                    phi_274_ = type_28_(_e203_, _e201_, _e205_, _e201_, true, true);
                    phi_275_ = type_28_();
                    phi_276_ = true;
                } else {
                    let _e120_ = (_e91_ - _e114_);
                    if (_e100_ == 0u) {
                        phi_267_ = type_28_();
                        phi_268_ = true;
                    } else {
                        let _e122_ = (_e120_ >= _e115_);
                        if _e122_ {
                            phi_266_ = type_28_();
                        } else {
                            phi_239_ = (_e97_ - 1u);
                            phi_242_ = 0u;
                            loop {
                                let _e125_ = phi_239_;
                                let _e127_ = phi_242_;
                                local_2_ = _e127_;
                                local_3_ = _e127_;
                                local_4_ = _e127_;
                                let _e128_ = (_e127_ < _e125_);
                                if _e128_ {
                                    let _e133_ = (_e127_ + (((_e125_ - _e127_) + 1u) >> bitcast<u32>(1i)));
                                    let _e135_ = ((_e133_ * _e100_) <= _e120_);
                                    if _e135_ {
                                        phi_260_ = _e125_;
                                    } else {
                                        phi_260_ = (_e133_ - 1u);
                                    }
                                    let _e138_ = phi_260_;
                                    phi_240_ = _e138_;
                                    phi_243_ = select(_e127_, _e133_, _e135_);
                                } else {
                                    phi_240_ = u32();
                                    phi_243_ = u32();
                                }
                                let _e141_ = phi_240_;
                                let _e143_ = phi_243_;
                                phi_239_ = _e141_;
                                phi_242_ = _e143_;
                                if !(_e128_) {
                                    break;
                                }
                            }
                            let _e146_ = local_2_;
                            let _e148_ = (_e120_ - (_e146_ * _e100_));
                            let _e151_ = local_3_;
                            let _e153_ = local_4_;
                            phi_266_ = type_28_(_e151_, _e148_, _e153_, (_e97_ + _e148_), true, false);
                        }
                        let _e156_ = phi_266_;
                        phi_267_ = _e156_;
                        phi_268_ = _e122_;
                    }
                    let _e158_ = phi_267_;
                    let _e160_ = phi_268_;
                    if _e160_ {
                        phi_273_ = type_28_(0u, 0u, 0u, 0u, false, false);
                    } else {
                        phi_273_ = type_28_();
                    }
                    let _e162_ = phi_273_;
                    phi_274_ = _e162_;
                    phi_275_ = _e158_;
                    phi_276_ = _e160_;
                }
                let _e208_ = phi_274_;
                let _e210_ = phi_275_;
                let _e212_ = phi_276_;
                if _e212_ {
                    phi_280_ = _e208_;
                } else {
                    phi_280_ = _e210_;
                }
                let _e214_ = phi_280_;
                phi_281_ = _e214_;
            }
            let _e216_ = phi_281_;
            let _e219_ = uniforms.member.member_6_;
            if (_e219_ == 0u) {
                let _e577_ = (_e79_ == 0u);
                if _e577_ {
                    phi_300_ = 0u;
                } else {
                    let _e578_ = (_e79_ - 1u);
                    phi_300_ = select(_e578_, _e216_.member_2_, (_e216_.member_2_ < _e578_));
                }
                let _e582_ = phi_300_;
                if (_e582_ < _e79_) {
                } else {
                    break;
                }
                let _e587_ = col_scales.member[_e582_].x;
                let _e591_ = col_scales.member[_e582_].y;
                if _e577_ {
                    phi_322_ = 0u;
                } else {
                    let _e593_ = (_e79_ - 1u);
                    phi_322_ = select(_e593_, _e216_.member_3_, (_e216_.member_3_ < _e593_));
                }
                let _e597_ = phi_322_;
                if (_e597_ < _e79_) {
                } else {
                    break;
                }
                let _e602_ = col_scales.member[_e597_].x;
                let _e606_ = col_scales.member[_e597_].y;
                let _e607_ = bitcast<u32>(_e591_);
                let _e610_ = ((_e607_ >> bitcast<u32>(23u)) & 255u);
                if ((_e607_ & 2155872255u) == 0u) {
                    let _e613_ = (_e610_ > 0u);
                    if _e613_ {
                        let _e615_ = bitcast<u32>(_e587_);
                        let _e617_ = bitcast<f32>((_e615_ & 2147483647u));
                        if (_e610_ < 255u) {
                            if ((_e615_ & 2139095040u) != 2139095040u) {
                                phi_364_ = select(true, false, ((_e617_ * _e591_) <= 1f));
                            } else {
                                phi_364_ = true;
                            }
                            let _e624_ = phi_364_;
                            phi_365_ = _e624_;
                        } else {
                            phi_365_ = true;
                        }
                        let _e626_ = phi_365_;
                        phi_366_ = _e617_;
                        phi_368_ = _e626_;
                    } else {
                        phi_366_ = f32();
                        phi_368_ = false;
                    }
                    let _e628_ = phi_366_;
                    let _e630_ = phi_368_;
                    phi_369_ = _e628_;
                    phi_370_ = select(true, false, _e613_);
                    phi_371_ = _e630_;
                } else {
                    phi_369_ = f32();
                    phi_370_ = true;
                    phi_371_ = false;
                }
                let _e633_ = phi_369_;
                let _e635_ = phi_370_;
                let _e637_ = phi_371_;
                if _e635_ {
                    phi_378_ = bitcast<f32>((bitcast<u32>(_e587_) & 2147483647u));
                } else {
                    phi_378_ = _e633_;
                }
                let _e642_ = phi_378_;
                if select(_e637_, true, _e635_) {
                    let _e644_ = bitcast<u32>(_e642_);
                    if ((_e644_ & 2147483648u) == 0u) {
                        if (_e644_ == 0u) {
                            phi_423_ = type_14_();
                            phi_424_ = true;
                        } else {
                            let _e648_ = (_e644_ >= 2139095040u);
                            if _e648_ {
                                phi_422_ = type_14_();
                            } else {
                                if (_e642_ != _e642_) {
                                    phi_1908_ = true;
                                } else {
                                    phi_1908_ = (85070590000000000000000000000000000000f <= _e642_);
                                }
                                let _e652_ = phi_1908_;
                                let _e658_ = (((bitcast<u32>(select(_e642_, 85070590000000000000000000000000000000f, _e652_)) + 8388607u) & 2139095040u) >> bitcast<u32>(23u));
                                if (_e658_ < 1u) {
                                    phi_415_ = 1u;
                                } else {
                                    phi_415_ = select(_e658_, 253u, (_e658_ > 253u));
                                }
                                let _e663_ = phi_415_;
                                phi_422_ = type_14_(bitcast<f32>((_e663_ << bitcast<u32>(23u))), bitcast<f32>(((254u - _e663_) << bitcast<u32>(23u))));
                            }
                            let _e673_ = phi_422_;
                            phi_423_ = _e673_;
                            phi_424_ = _e648_;
                        }
                        let _e675_ = phi_423_;
                        let _e677_ = phi_424_;
                        phi_425_ = _e675_;
                        phi_426_ = _e677_;
                    } else {
                        phi_425_ = type_14_();
                        phi_426_ = true;
                    }
                    let _e679_ = phi_425_;
                    let _e681_ = phi_426_;
                    if _e681_ {
                        phi_431_ = type_14_(1f, 1f);
                    } else {
                        phi_431_ = _e679_;
                    }
                    let _e683_ = phi_431_;
                    phi_433_ = _e683_.member_1_;
                } else {
                    phi_433_ = _e591_;
                }
                let _e686_ = phi_433_;
                let _e687_ = bitcast<u32>(_e606_);
                let _e690_ = ((_e687_ >> bitcast<u32>(23u)) & 255u);
                if ((_e687_ & 2155872255u) == 0u) {
                    let _e693_ = (_e690_ > 0u);
                    if _e693_ {
                        let _e695_ = bitcast<u32>(_e602_);
                        let _e697_ = bitcast<f32>((_e695_ & 2147483647u));
                        if (_e690_ < 255u) {
                            if ((_e695_ & 2139095040u) != 2139095040u) {
                                phi_465_ = select(true, false, ((_e697_ * _e606_) <= 1f));
                            } else {
                                phi_465_ = true;
                            }
                            let _e704_ = phi_465_;
                            phi_466_ = _e704_;
                        } else {
                            phi_466_ = true;
                        }
                        let _e706_ = phi_466_;
                        phi_467_ = _e697_;
                        phi_469_ = _e706_;
                    } else {
                        phi_467_ = f32();
                        phi_469_ = false;
                    }
                    let _e708_ = phi_467_;
                    let _e710_ = phi_469_;
                    phi_470_ = _e708_;
                    phi_471_ = select(true, false, _e693_);
                    phi_472_ = _e710_;
                } else {
                    phi_470_ = f32();
                    phi_471_ = true;
                    phi_472_ = false;
                }
                let _e713_ = phi_470_;
                let _e715_ = phi_471_;
                let _e717_ = phi_472_;
                if _e715_ {
                    phi_479_ = bitcast<f32>((bitcast<u32>(_e602_) & 2147483647u));
                } else {
                    phi_479_ = _e713_;
                }
                let _e722_ = phi_479_;
                if select(_e717_, true, _e715_) {
                    let _e724_ = bitcast<u32>(_e722_);
                    if ((_e724_ & 2147483648u) == 0u) {
                        if (_e724_ == 0u) {
                            phi_524_ = type_14_();
                            phi_525_ = true;
                        } else {
                            let _e728_ = (_e724_ >= 2139095040u);
                            if _e728_ {
                                phi_523_ = type_14_();
                            } else {
                                if (_e722_ != _e722_) {
                                    phi_1923_ = true;
                                } else {
                                    phi_1923_ = (85070590000000000000000000000000000000f <= _e722_);
                                }
                                let _e732_ = phi_1923_;
                                let _e738_ = (((bitcast<u32>(select(_e722_, 85070590000000000000000000000000000000f, _e732_)) + 8388607u) & 2139095040u) >> bitcast<u32>(23u));
                                if (_e738_ < 1u) {
                                    phi_516_ = 1u;
                                } else {
                                    phi_516_ = select(_e738_, 253u, (_e738_ > 253u));
                                }
                                let _e743_ = phi_516_;
                                phi_523_ = type_14_(bitcast<f32>((_e743_ << bitcast<u32>(23u))), bitcast<f32>(((254u - _e743_) << bitcast<u32>(23u))));
                            }
                            let _e753_ = phi_523_;
                            phi_524_ = _e753_;
                            phi_525_ = _e728_;
                        }
                        let _e755_ = phi_524_;
                        let _e757_ = phi_525_;
                        phi_526_ = _e755_;
                        phi_527_ = _e757_;
                    } else {
                        phi_526_ = type_14_();
                        phi_527_ = true;
                    }
                    let _e759_ = phi_526_;
                    let _e761_ = phi_527_;
                    if _e761_ {
                        phi_532_ = type_14_(1f, 1f);
                    } else {
                        phi_532_ = _e759_;
                    }
                    let _e763_ = phi_532_;
                    phi_534_ = _e763_.member_1_;
                } else {
                    phi_534_ = _e606_;
                }
                let _e766_ = phi_534_;
                if _e216_.member_4_ {
                    phi_541_ = (_e72_.z < _e110_);
                } else {
                    phi_541_ = false;
                }
                let _e771_ = phi_541_;
                phi_544_ = type_12_(0u, 4u);
                phi_547_ = array<i32, 6>(0i, 0i, 0i, 0i, 0i, 0i);
                loop {
                    let _e773_ = phi_544_;
                    let _e775_ = phi_547_;
                    local_13_ = _e775_;
                    local_14_ = _e775_;
                    local_15_ = _e775_;
                    local_16_ = _e775_;
                    local_17_ = _e775_;
                    if (_e773_.member < _e773_.member_1_) {
                        phi_545_ = type_12_((_e773_.member + 1u), _e773_.member_1_);
                        phi_570_ = type_12_(1u, _e773_.member);
                    } else {
                        phi_545_ = _e773_;
                        phi_570_ = type_12_(0u, u32());
                    }
                    let _e788_ = phi_545_;
                    let _e790_ = phi_570_;
                    let _e794_ = (bitcast<i32>(_e790_.member) != 0i);
                    if _e794_ {
                        let _e796_ = ((_e790_.member_1_ * 256u) + _e82_);
                        let _e797_ = ((_e72_.z * _e105_) + _e796_);
                        if _e771_ {
                            if (_e796_ < _e105_) {
                                if (_e797_ < _e94_) {
                                    let _e802_ = ((_e216_.member * _e94_) + _e797_);
                                    let _e803_ = (_e75_ == 0u);
                                    if _e803_ {
                                        phi_605_ = 0u;
                                    } else {
                                        let _e804_ = (_e75_ - 1u);
                                        phi_605_ = select(_e804_, _e802_, (_e802_ < _e804_));
                                    }
                                    let _e808_ = phi_605_;
                                    if (_e808_ < _e75_) {
                                    } else {
                                        phi_1966_ = true;
                                        break;
                                    }
                                    let _e812_ = features.member[_e808_];
                                    if _e216_.member_5_ {
                                        let _e829_ = ((_e216_.member_1_ * _e94_) + _e797_);
                                        if _e803_ {
                                            phi_630_ = 0u;
                                        } else {
                                            let _e830_ = (_e75_ - 1u);
                                            phi_630_ = select(_e830_, _e829_, (_e829_ < _e830_));
                                        }
                                        let _e834_ = phi_630_;
                                        if (_e834_ < _e75_) {
                                        } else {
                                            phi_1966_ = true;
                                            break;
                                        }
                                        let _e838_ = features.member[_e834_];
                                        phi_659_ = _e838_;
                                    } else {
                                        let _e816_ = ((_e216_.member_1_ * _e94_) + _e797_);
                                        if (_e77_ == 0u) {
                                            phi_651_ = 0u;
                                        } else {
                                            let _e818_ = (_e77_ - 1u);
                                            phi_651_ = select(_e818_, _e816_, (_e816_ < _e818_));
                                        }
                                        let _e822_ = phi_651_;
                                        if (_e822_ < _e77_) {
                                        } else {
                                            phi_1966_ = true;
                                            break;
                                        }
                                        let _e826_ = targets.member[_e822_];
                                        phi_659_ = _e826_;
                                    }
                                    let _e840_ = phi_659_;
                                    if ((bitcast<u32>(_e812_) & 2139095040u) != 2139095040u) {
                                        let _e846_ = ((bitcast<u32>(_e686_) & 2139095040u) != 2139095040u);
                                        if _e846_ {
                                            let _e848_ = select(_e812_, -85070590000000000000000000000000000000f, (_e812_ < -85070590000000000000000000000000000000f));
                                            let _e851_ = (select(_e848_, 85070590000000000000000000000000000000f, (_e848_ > 85070590000000000000000000000000000000f)) * _e686_);
                                            if ((bitcast<u32>(_e851_) & 2139095040u) != 2139095040u) {
                                                let _e856_ = select(_e851_, -1f, (_e851_ < -1f));
                                                let _e858_ = select(_e856_, 1f, (_e856_ > 1f));
                                                let _e859_ = (_e858_ * 256f);
                                                if ((bitcast<u32>(_e859_) & 2139095040u) != 2139095040u) {
                                                    let _e864_ = select(_e859_, -4194304f, (_e859_ < -4194304f));
                                                    phi_734_ = (bitcast<i32>(bitcast<u32>((select(_e864_, 4194304f, (_e864_ > 4194304f)) + 12582912f))) - 1262485504i);
                                                } else {
                                                    phi_734_ = 0i;
                                                }
                                                let _e872_ = phi_734_;
                                                let _e875_ = (_e858_ - (f32(_e872_) * 0.00390625f));
                                                let _e876_ = (_e875_ * 65536f);
                                                if ((bitcast<u32>(_e876_) & 2139095040u) != 2139095040u) {
                                                    let _e881_ = select(_e876_, -4194304f, (_e876_ < -4194304f));
                                                    phi_764_ = (bitcast<i32>(bitcast<u32>((select(_e881_, 4194304f, (_e881_ > 4194304f)) + 12582912f))) - 1262485504i);
                                                } else {
                                                    phi_764_ = 0i;
                                                }
                                                let _e889_ = phi_764_;
                                                let _e893_ = ((_e875_ - (f32(_e889_) * 0.000015258789f)) * 16777216f);
                                                if ((bitcast<u32>(_e893_) & 2139095040u) != 2139095040u) {
                                                    let _e898_ = select(_e893_, -4194304f, (_e893_ < -4194304f));
                                                    phi_794_ = (bitcast<i32>(bitcast<u32>((select(_e898_, 4194304f, (_e898_ > 4194304f)) + 12582912f))) - 1262485504i);
                                                } else {
                                                    phi_794_ = 0i;
                                                }
                                                let _e906_ = phi_794_;
                                                phi_797_ = array<i32, 3>(_e872_, _e889_, _e906_);
                                            } else {
                                                phi_797_ = array<i32, 3>(0i, 0i, 0i);
                                            }
                                            let _e909_ = phi_797_;
                                            phi_798_ = _e909_;
                                        } else {
                                            phi_798_ = array<i32, 3>();
                                        }
                                        let _e911_ = phi_798_;
                                        phi_800_ = _e911_;
                                        phi_801_ = select(true, false, _e846_);
                                    } else {
                                        phi_800_ = array<i32, 3>();
                                        phi_801_ = true;
                                    }
                                    let _e914_ = phi_800_;
                                    let _e916_ = phi_801_;
                                    if _e916_ {
                                        phi_806_ = array<i32, 3>(0i, 0i, 0i);
                                    } else {
                                        phi_806_ = _e914_;
                                    }
                                    let _e918_ = phi_806_;
                                    if ((bitcast<u32>(_e840_) & 2139095040u) != 2139095040u) {
                                        let _e924_ = ((bitcast<u32>(_e766_) & 2139095040u) != 2139095040u);
                                        if _e924_ {
                                            let _e926_ = select(_e840_, -85070590000000000000000000000000000000f, (_e840_ < -85070590000000000000000000000000000000f));
                                            let _e929_ = (select(_e926_, 85070590000000000000000000000000000000f, (_e926_ > 85070590000000000000000000000000000000f)) * _e766_);
                                            if ((bitcast<u32>(_e929_) & 2139095040u) != 2139095040u) {
                                                let _e934_ = select(_e929_, -1f, (_e929_ < -1f));
                                                let _e936_ = select(_e934_, 1f, (_e934_ > 1f));
                                                let _e937_ = (_e936_ * 256f);
                                                if ((bitcast<u32>(_e937_) & 2139095040u) != 2139095040u) {
                                                    let _e942_ = select(_e937_, -4194304f, (_e937_ < -4194304f));
                                                    phi_881_ = (bitcast<i32>(bitcast<u32>((select(_e942_, 4194304f, (_e942_ > 4194304f)) + 12582912f))) - 1262485504i);
                                                } else {
                                                    phi_881_ = 0i;
                                                }
                                                let _e950_ = phi_881_;
                                                let _e953_ = (_e936_ - (f32(_e950_) * 0.00390625f));
                                                let _e954_ = (_e953_ * 65536f);
                                                if ((bitcast<u32>(_e954_) & 2139095040u) != 2139095040u) {
                                                    let _e959_ = select(_e954_, -4194304f, (_e954_ < -4194304f));
                                                    phi_911_ = (bitcast<i32>(bitcast<u32>((select(_e959_, 4194304f, (_e959_ > 4194304f)) + 12582912f))) - 1262485504i);
                                                } else {
                                                    phi_911_ = 0i;
                                                }
                                                let _e967_ = phi_911_;
                                                let _e971_ = ((_e953_ - (f32(_e967_) * 0.000015258789f)) * 16777216f);
                                                if ((bitcast<u32>(_e971_) & 2139095040u) != 2139095040u) {
                                                    let _e976_ = select(_e971_, -4194304f, (_e971_ < -4194304f));
                                                    phi_941_ = (bitcast<i32>(bitcast<u32>((select(_e976_, 4194304f, (_e976_ > 4194304f)) + 12582912f))) - 1262485504i);
                                                } else {
                                                    phi_941_ = 0i;
                                                }
                                                let _e984_ = phi_941_;
                                                phi_944_ = array<i32, 3>(_e950_, _e967_, _e984_);
                                            } else {
                                                phi_944_ = array<i32, 3>(0i, 0i, 0i);
                                            }
                                            let _e987_ = phi_944_;
                                            phi_945_ = _e987_;
                                        } else {
                                            phi_945_ = array<i32, 3>();
                                        }
                                        let _e989_ = phi_945_;
                                        phi_947_ = _e989_;
                                        phi_948_ = select(true, false, _e924_);
                                    } else {
                                        phi_947_ = array<i32, 3>();
                                        phi_948_ = true;
                                    }
                                    let _e992_ = phi_947_;
                                    let _e994_ = phi_948_;
                                    if _e994_ {
                                        phi_953_ = array<i32, 3>(0i, 0i, 0i);
                                    } else {
                                        phi_953_ = _e992_;
                                    }
                                    let _e996_ = phi_953_;
                                    let _e1008_ = array<i32, 6>((_e775_[0] + (_e918_[0] * _e996_[0])), _e775_[1], _e775_[2], _e775_[3], _e775_[4], _e775_[5]);
                                    let _e1022_ = array<i32, 6>(_e1008_[0], (_e775_[1] + ((_e918_[0] * _e996_[1]) + (_e918_[1] * _e996_[0]))), _e1008_[2], _e1008_[3], _e1008_[4], _e1008_[5]);
                                    let _e1038_ = array<i32, 6>(_e1022_[0], _e1022_[1], (_e775_[2] + (((_e918_[0] * _e996_[2]) + (_e918_[1] * _e996_[1])) + (_e918_[2] * _e996_[0]))), _e1022_[3], _e1022_[4], _e1022_[5]);
                                    let _e1050_ = array<i32, 6>(_e1038_[0], _e1038_[1], _e1038_[2], (_e775_[3] + ((_e918_[1] * _e996_[2]) + (_e918_[2] * _e996_[1]))), _e1038_[4], _e1038_[5]);
                                    phi_1000_ = array<i32, 6>(_e1050_[0], _e1050_[1], _e1050_[2], _e1050_[3], (_e775_[4] + (_e918_[2] * _e996_[2])), _e1050_[5]);
                                } else {
                                    phi_1000_ = _e775_;
                                }
                                let _e1062_ = phi_1000_;
                                phi_1001_ = _e1062_;
                            } else {
                                phi_1001_ = _e775_;
                            }
                            let _e1064_ = phi_1001_;
                            phi_1002_ = _e1064_;
                        } else {
                            phi_1002_ = _e775_;
                        }
                        let _e1066_ = phi_1002_;
                        phi_548_ = _e1066_;
                    } else {
                        phi_548_ = array<i32, 6>();
                    }
                    let _e1068_ = phi_548_;
                    phi_544_ = _e788_;
                    phi_547_ = _e1068_;
                    phi_1966_ = false;
                    if !(_e794_) {
                        break;
                    }
                }
                let _e1071_ = phi_1966_;
                if _e1071_ {
                    break;
                }
                let _e1073_ = local_13_;
                let _e1076_ = local_14_;
                let _e1079_ = local_15_;
                let _e1082_ = local_16_;
                let _e1085_ = local_17_;
                local = array<i32, 5>(_e1073_[0], _e1076_[1], _e1079_[2], _e1082_[3], _e1085_[4]);
                phi_1017_ = type_12_(0u, 5u);
                loop {
                    let _e1089_ = phi_1017_;
                    if (_e1089_.member < _e1089_.member_1_) {
                        phi_1018_ = type_12_((_e1089_.member + 1u), _e1089_.member_1_);
                        phi_1041_ = type_12_(1u, _e1089_.member);
                    } else {
                        phi_1018_ = _e1089_;
                        phi_1041_ = type_12_(0u, u32());
                    }
                    let _e1102_ = phi_1018_;
                    let _e1104_ = phi_1041_;
                    let _e1108_ = (bitcast<i32>(_e1104_.member) != 0i);
                    if _e1108_ {
                        if (_e1104_.member_1_ < 5u) {
                        } else {
                            phi_2049_ = true;
                            break;
                        }
                        let _e1111_ = local[_e1104_.member_1_];
                        let _e1113_ = ((_e1104_.member_1_ * 256u) + _e82_);
                        let _e1115_ = select(1279u, _e1113_, (_e1113_ < 1279u));
                        if (_e1115_ < 1280u) {
                        } else {
                            phi_2049_ = true;
                            break;
                        }
                        reduce_i32_[_e1115_] = _e1111_;
                    }
                    phi_1017_ = _e1102_;
                    phi_2049_ = _e1071_;
                    if !(_e1108_) {
                        break;
                    }
                }
                let _e1120_ = phi_2049_;
                if _e1120_ {
                    break;
                }
                workgroupBarrier();
                phi_2059_ = _e1120_;
                phi_1076_ = 128u;
                loop {
                    let _e1122_ = phi_2059_;
                    let _e1124_ = phi_1076_;
                    let _e1125_ = (_e1124_ > 0u);
                    if _e1125_ {
                        if (_e82_ < _e1124_) {
                            phi_1091_ = type_12_(0u, 5u);
                            loop {
                                let _e1128_ = phi_1091_;
                                if (_e1128_.member < _e1128_.member_1_) {
                                    phi_1092_ = type_12_((_e1128_.member + 1u), _e1128_.member_1_);
                                    phi_1115_ = type_12_(1u, _e1128_.member);
                                } else {
                                    phi_1092_ = _e1128_;
                                    phi_1115_ = type_12_(0u, u32());
                                }
                                let _e1141_ = phi_1092_;
                                let _e1143_ = phi_1115_;
                                let _e1147_ = (bitcast<i32>(_e1143_.member) != 0i);
                                if _e1147_ {
                                    let _e1149_ = ((_e1143_.member_1_ * 256u) + _e82_);
                                    let _e1150_ = (_e1149_ < 1280u);
                                    if _e1150_ {
                                    } else {
                                        phi_2055_ = true;
                                        break;
                                    }
                                    let _e1152_ = reduce_i32_[_e1149_];
                                    let _e1153_ = (_e1149_ + _e1124_);
                                    if (_e1153_ < 1280u) {
                                    } else {
                                        phi_2055_ = true;
                                        break;
                                    }
                                    let _e1156_ = reduce_i32_[_e1153_];
                                    if _e1150_ {
                                    } else {
                                        phi_2055_ = true;
                                        break;
                                    }
                                    reduce_i32_[_e1149_] = (_e1152_ + _e1156_);
                                }
                                phi_1091_ = _e1141_;
                                phi_2055_ = _e1122_;
                                if !(_e1147_) {
                                    break;
                                }
                            }
                            let _e1160_ = phi_2055_;
                            phi_2060_ = _e1160_;
                            if _e1160_ {
                                break;
                            }
                            phi_2062_ = _e1160_;
                        } else {
                            phi_2062_ = _e1122_;
                        }
                        let _e1162_ = phi_2062_;
                        workgroupBarrier();
                        phi_2061_ = _e1162_;
                        phi_1077_ = (_e1124_ >> bitcast<u32>(1i));
                    } else {
                        phi_2061_ = _e1122_;
                        phi_1077_ = u32();
                    }
                    let _e1166_ = phi_2061_;
                    let _e1168_ = phi_1077_;
                    phi_2059_ = _e1166_;
                    phi_1076_ = _e1168_;
                    phi_2060_ = _e1166_;
                    if !(_e1125_) {
                        break;
                    }
                }
                let _e1171_ = phi_2060_;
                if _e1171_ {
                    break;
                }
                let _e1173_ = reduce_i32_[0];
                let _e1175_ = reduce_i32_[256];
                let _e1177_ = reduce_i32_[512];
                let _e1179_ = reduce_i32_[768];
                let _e1181_ = reduce_i32_[1024];
                workgroupBarrier();
                local_1_ = array<i32, 5>(_e1173_, _e1175_, _e1177_, _e1179_, _e1181_);
                if (_e82_ == 0u) {
                    if _e771_ {
                        let _e1187_ = (_e117_ + (((_e91_ * _e110_) + _e72_.z) * 6u));
                        phi_1180_ = type_12_(0u, 5u);
                        loop {
                            let _e1189_ = phi_1180_;
                            if (_e1189_.member < _e1189_.member_1_) {
                                phi_1181_ = type_12_((_e1189_.member + 1u), _e1189_.member_1_);
                                phi_1204_ = type_12_(1u, _e1189_.member);
                            } else {
                                phi_1181_ = _e1189_;
                                phi_1204_ = type_12_(0u, u32());
                            }
                            let _e1202_ = phi_1181_;
                            let _e1204_ = phi_1204_;
                            let _e1208_ = (bitcast<i32>(_e1204_.member) != 0i);
                            if _e1208_ {
                                if (_e1204_.member_1_ < 5u) {
                                } else {
                                    phi_2063_ = true;
                                    break;
                                }
                                let _e1211_ = local_1_[_e1204_.member_1_];
                                let _e1212_ = (_e1187_ + _e1204_.member_1_);
                                if (_e81_ == 0u) {
                                    phi_1230_ = 0u;
                                } else {
                                    let _e1214_ = (_e81_ - 1u);
                                    phi_1230_ = select(_e1214_, _e1212_, (_e1212_ < _e1214_));
                                }
                                let _e1218_ = phi_1230_;
                                if (_e1218_ < _e81_) {
                                } else {
                                    phi_2063_ = true;
                                    break;
                                }
                                ozaki_buckets.member[_e1218_] = _e1211_;
                            }
                            phi_1180_ = _e1202_;
                            phi_2063_ = _e1171_;
                            if !(_e1208_) {
                                break;
                            }
                        }
                        let _e1224_ = phi_2063_;
                        if _e1224_ {
                            break;
                        }
                        let _e1225_ = (_e1187_ + 5u);
                        if (_e81_ == 0u) {
                            phi_1248_ = 0u;
                        } else {
                            let _e1227_ = (_e81_ - 1u);
                            phi_1248_ = select(_e1227_, _e1225_, (_e1225_ < _e1227_));
                        }
                        let _e1231_ = phi_1248_;
                        if (_e1231_ < _e81_) {
                        } else {
                            break;
                        }
                        ozaki_buckets.member[_e1231_] = 0i;
                    }
                }
            } else {
                phi_1259_ = _e82_;
                phi_1262_ = type_12_(0u, 0u);
                phi_1264_ = type_12_(0u, 0u);
                loop {
                    let _e223_ = phi_1259_;
                    let _e225_ = phi_1262_;
                    let _e227_ = phi_1264_;
                    local_9_ = _e227_;
                    local_10_ = _e227_;
                    local_11_ = _e225_;
                    local_12_ = _e225_;
                    let _e228_ = (_e223_ < _e110_);
                    if _e228_ {
                        if _e216_.member_4_ {
                            let _e232_ = (_e117_ + (((_e91_ * _e110_) + _e223_) * 6u));
                            let _e233_ = (_e81_ == 0u);
                            if _e233_ {
                                phi_1289_ = 0u;
                            } else {
                                let _e234_ = (_e81_ - 1u);
                                phi_1289_ = select(_e234_, _e232_, (_e232_ < _e234_));
                            }
                            let _e238_ = phi_1289_;
                            if (_e238_ < _e81_) {
                            } else {
                                phi_1928_ = true;
                                break;
                            }
                            let _e242_ = ozaki_buckets.member[_e238_];
                            let _e243_ = (_e232_ + 1u);
                            if _e233_ {
                                phi_1308_ = 0u;
                            } else {
                                let _e244_ = (_e81_ - 1u);
                                phi_1308_ = select(_e244_, _e243_, (_e243_ < _e244_));
                            }
                            let _e248_ = phi_1308_;
                            if (_e248_ < _e81_) {
                            } else {
                                phi_1928_ = true;
                                break;
                            }
                            let _e252_ = ozaki_buckets.member[_e248_];
                            let _e253_ = (_e232_ + 2u);
                            if _e233_ {
                                phi_1327_ = 0u;
                            } else {
                                let _e254_ = (_e81_ - 1u);
                                phi_1327_ = select(_e254_, _e253_, (_e253_ < _e254_));
                            }
                            let _e258_ = phi_1327_;
                            if (_e258_ < _e81_) {
                            } else {
                                phi_1928_ = true;
                                break;
                            }
                            let _e262_ = ozaki_buckets.member[_e258_];
                            let _e263_ = (_e232_ + 3u);
                            if _e233_ {
                                phi_1346_ = 0u;
                            } else {
                                let _e264_ = (_e81_ - 1u);
                                phi_1346_ = select(_e264_, _e263_, (_e263_ < _e264_));
                            }
                            let _e268_ = phi_1346_;
                            if (_e268_ < _e81_) {
                            } else {
                                phi_1928_ = true;
                                break;
                            }
                            let _e272_ = ozaki_buckets.member[_e268_];
                            let _e273_ = (_e232_ + 4u);
                            if _e233_ {
                                phi_1365_ = 0u;
                            } else {
                                let _e274_ = (_e81_ - 1u);
                                phi_1365_ = select(_e274_, _e273_, (_e273_ < _e274_));
                            }
                            let _e278_ = phi_1365_;
                            if (_e278_ < _e81_) {
                            } else {
                                phi_1928_ = true;
                                break;
                            }
                            let _e282_ = ozaki_buckets.member[_e278_];
                            let _e283_ = bitcast<u32>(_e242_);
                            let _e296_ = (_e227_.member + (_e283_ << bitcast<u32>(16u)));
                            let _e301_ = bitcast<u32>(_e252_);
                            let _e312_ = (_e296_ + (_e301_ << bitcast<u32>(8u)));
                            let _e321_ = (_e312_ + bitcast<u32>(_e262_));
                            let _e327_ = bitcast<u32>(_e272_);
                            let _e340_ = (_e225_.member + (_e327_ << bitcast<u32>(8u)));
                            let _e349_ = (_e340_ + bitcast<u32>(_e282_));
                            phi_1504_ = type_12_(_e349_, ((((_e225_.member_1_ + ((bitcast<u32>((_e272_ >> bitcast<u32>(31i))) << bitcast<u32>(8u)) | (_e327_ >> bitcast<u32>(24u)))) + select(0u, 1u, (_e340_ < _e225_.member))) + bitcast<u32>((_e282_ >> bitcast<u32>(31i)))) + select(0u, 1u, (_e349_ < _e340_))));
                            phi_1505_ = type_12_(_e321_, ((((((_e227_.member_1_ + ((bitcast<u32>((_e242_ >> bitcast<u32>(31i))) << bitcast<u32>(16u)) | (_e283_ >> bitcast<u32>(16u)))) + select(0u, 1u, (_e296_ < _e227_.member))) + ((bitcast<u32>((_e252_ >> bitcast<u32>(31i))) << bitcast<u32>(8u)) | (_e301_ >> bitcast<u32>(24u)))) + select(0u, 1u, (_e312_ < _e296_))) + bitcast<u32>((_e262_ >> bitcast<u32>(31i)))) + select(0u, 1u, (_e321_ < _e312_))));
                        } else {
                            phi_1504_ = _e225_;
                            phi_1505_ = _e227_;
                        }
                        let _e356_ = phi_1504_;
                        let _e358_ = phi_1505_;
                        phi_1260_ = (_e223_ + 256u);
                        phi_1263_ = _e356_;
                        phi_1265_ = _e358_;
                    } else {
                        phi_1260_ = u32();
                        phi_1263_ = type_12_();
                        phi_1265_ = type_12_();
                    }
                    let _e361_ = phi_1260_;
                    let _e363_ = phi_1263_;
                    let _e365_ = phi_1265_;
                    phi_1259_ = _e361_;
                    phi_1262_ = _e363_;
                    phi_1264_ = _e365_;
                    phi_1928_ = false;
                    if !(_e228_) {
                        break;
                    }
                }
                let _e368_ = phi_1928_;
                if _e368_ {
                    break;
                }
                let _e370_ = local_9_;
                let _e373_ = local_10_;
                let _e376_ = local_11_;
                let _e379_ = local_12_;
                let _e381_ = (_e82_ & 255u);
                let _e382_ = (_e381_ < 1280u);
                if _e382_ {
                } else {
                    break;
                }
                reduce_i32_[_e381_] = bitcast<i32>(_e370_.member);
                let _e385_ = (256u + _e381_);
                let _e386_ = (_e385_ < 1280u);
                if _e386_ {
                } else {
                    break;
                }
                reduce_i32_[_e385_] = bitcast<i32>(_e373_.member_1_);
                let _e389_ = (512u + _e381_);
                let _e390_ = (_e389_ < 1280u);
                if _e390_ {
                } else {
                    break;
                }
                reduce_i32_[_e389_] = bitcast<i32>(_e376_.member);
                let _e393_ = (768u + _e381_);
                let _e394_ = (_e393_ < 1280u);
                if _e394_ {
                } else {
                    break;
                }
                reduce_i32_[_e393_] = bitcast<i32>(_e379_.member_1_);
                workgroupBarrier();
                phi_1544_ = 128u;
                loop {
                    let _e398_ = phi_1544_;
                    let _e399_ = (_e398_ > 0u);
                    if _e399_ {
                        if (_e381_ < _e398_) {
                            let _e401_ = (_e381_ + _e398_);
                            if _e382_ {
                            } else {
                                phi_1962_ = true;
                                break;
                            }
                            let _e402_ = reduce_i32_[_e381_];
                            let _e403_ = bitcast<u32>(_e402_);
                            if _e386_ {
                            } else {
                                phi_1962_ = true;
                                break;
                            }
                            let _e404_ = reduce_i32_[_e385_];
                            if (_e401_ < 1280u) {
                            } else {
                                phi_1962_ = true;
                                break;
                            }
                            let _e408_ = reduce_i32_[_e401_];
                            let _e410_ = (256u + _e401_);
                            if (_e410_ < 1280u) {
                            } else {
                                phi_1962_ = true;
                                break;
                            }
                            let _e413_ = reduce_i32_[_e410_];
                            let _e415_ = (_e403_ + bitcast<u32>(_e408_));
                            if _e382_ {
                            } else {
                                phi_1962_ = true;
                                break;
                            }
                            reduce_i32_[_e381_] = bitcast<i32>(_e415_);
                            if _e386_ {
                            } else {
                                phi_1962_ = true;
                                break;
                            }
                            reduce_i32_[_e385_] = bitcast<i32>(((bitcast<u32>(_e404_) + bitcast<u32>(_e413_)) + select(0u, 1u, (_e415_ < _e403_))));
                            if _e390_ {
                            } else {
                                phi_1962_ = true;
                                break;
                            }
                            let _e422_ = reduce_i32_[_e389_];
                            let _e423_ = bitcast<u32>(_e422_);
                            if _e394_ {
                            } else {
                                phi_1962_ = true;
                                break;
                            }
                            let _e424_ = reduce_i32_[_e393_];
                            let _e426_ = (512u + _e401_);
                            if (_e426_ < 1280u) {
                            } else {
                                phi_1962_ = true;
                                break;
                            }
                            let _e429_ = reduce_i32_[_e426_];
                            let _e431_ = (768u + _e401_);
                            if (_e431_ < 1280u) {
                            } else {
                                phi_1962_ = true;
                                break;
                            }
                            let _e434_ = reduce_i32_[_e431_];
                            let _e436_ = (_e423_ + bitcast<u32>(_e429_));
                            if _e390_ {
                            } else {
                                phi_1962_ = true;
                                break;
                            }
                            reduce_i32_[_e389_] = bitcast<i32>(_e436_);
                            if _e394_ {
                            } else {
                                phi_1962_ = true;
                                break;
                            }
                            reduce_i32_[_e393_] = bitcast<i32>(((bitcast<u32>(_e424_) + bitcast<u32>(_e434_)) + select(0u, 1u, (_e436_ < _e423_))));
                        }
                        workgroupBarrier();
                        phi_1545_ = (_e398_ >> bitcast<u32>(1i));
                    } else {
                        phi_1545_ = u32();
                    }
                    let _e446_ = phi_1545_;
                    phi_1544_ = _e446_;
                    phi_1962_ = _e368_;
                    if !(_e399_) {
                        break;
                    }
                }
                let _e449_ = phi_1962_;
                if _e449_ {
                    break;
                }
                let _e451_ = reduce_i32_[0];
                let _e452_ = bitcast<u32>(_e451_);
                let _e454_ = reduce_i32_[256];
                let _e455_ = bitcast<u32>(_e454_);
                let _e457_ = reduce_i32_[512];
                let _e458_ = bitcast<u32>(_e457_);
                let _e460_ = reduce_i32_[768];
                let _e461_ = bitcast<u32>(_e460_);
                workgroupBarrier();
                if (_e82_ == 0u) {
                    if _e216_.member_4_ {
                        let _e465_ = (_e91_ * 4u);
                        let _e468_ = uniforms.member.member_7_;
                        if ((_e468_ & 1u) == 0u) {
                            phi_1798_ = type_12_(_e458_, _e461_);
                            phi_1799_ = type_12_(_e452_, _e455_);
                        } else {
                            let _e471_ = (_e81_ == 0u);
                            if _e471_ {
                                phi_1717_ = 0u;
                            } else {
                                let _e472_ = (_e81_ - 1u);
                                phi_1717_ = select(_e472_, _e465_, (_e465_ < _e472_));
                            }
                            let _e476_ = phi_1717_;
                            if (_e476_ < _e81_) {
                            } else {
                                break;
                            }
                            let _e480_ = ozaki_buckets.member[_e476_];
                            let _e482_ = (_e465_ + 1u);
                            if _e471_ {
                                phi_1737_ = 0u;
                            } else {
                                let _e483_ = (_e81_ - 1u);
                                phi_1737_ = select(_e483_, _e482_, (_e482_ < _e483_));
                            }
                            let _e487_ = phi_1737_;
                            if (_e487_ < _e81_) {
                            } else {
                                break;
                            }
                            let _e491_ = ozaki_buckets.member[_e487_];
                            let _e493_ = (_e465_ + 2u);
                            if _e471_ {
                                phi_1757_ = 0u;
                            } else {
                                let _e494_ = (_e81_ - 1u);
                                phi_1757_ = select(_e494_, _e493_, (_e493_ < _e494_));
                            }
                            let _e498_ = phi_1757_;
                            if (_e498_ < _e81_) {
                            } else {
                                break;
                            }
                            let _e502_ = ozaki_buckets.member[_e498_];
                            let _e504_ = (_e465_ + 3u);
                            if _e471_ {
                                phi_1777_ = 0u;
                            } else {
                                let _e505_ = (_e81_ - 1u);
                                phi_1777_ = select(_e505_, _e504_, (_e504_ < _e505_));
                            }
                            let _e509_ = phi_1777_;
                            if (_e509_ < _e81_) {
                            } else {
                                break;
                            }
                            let _e513_ = ozaki_buckets.member[_e509_];
                            let _e515_ = (_e452_ + bitcast<u32>(_e480_));
                            let _e521_ = (_e458_ + bitcast<u32>(_e502_));
                            phi_1798_ = type_12_(_e521_, ((_e461_ + bitcast<u32>(_e513_)) + select(0u, 1u, (_e521_ < _e458_))));
                            phi_1799_ = type_12_(_e515_, ((_e455_ + bitcast<u32>(_e491_)) + select(0u, 1u, (_e515_ < _e452_))));
                        }
                        let _e528_ = phi_1798_;
                        let _e530_ = phi_1799_;
                        let _e532_ = (_e81_ == 0u);
                        if _e532_ {
                            phi_1811_ = 0u;
                        } else {
                            let _e533_ = (_e81_ - 1u);
                            phi_1811_ = select(_e533_, _e465_, (_e465_ < _e533_));
                        }
                        let _e537_ = phi_1811_;
                        if (_e537_ < _e81_) {
                        } else {
                            break;
                        }
                        ozaki_buckets.member[_e537_] = bitcast<i32>(_e530_.member);
                        let _e543_ = (_e465_ + 1u);
                        if _e532_ {
                            phi_1831_ = 0u;
                        } else {
                            let _e544_ = (_e81_ - 1u);
                            phi_1831_ = select(_e544_, _e543_, (_e543_ < _e544_));
                        }
                        let _e548_ = phi_1831_;
                        if (_e548_ < _e81_) {
                        } else {
                            break;
                        }
                        ozaki_buckets.member[_e548_] = bitcast<i32>(_e530_.member_1_);
                        let _e554_ = (_e465_ + 2u);
                        if _e532_ {
                            phi_1851_ = 0u;
                        } else {
                            let _e555_ = (_e81_ - 1u);
                            phi_1851_ = select(_e555_, _e554_, (_e554_ < _e555_));
                        }
                        let _e559_ = phi_1851_;
                        if (_e559_ < _e81_) {
                        } else {
                            break;
                        }
                        ozaki_buckets.member[_e559_] = bitcast<i32>(_e528_.member);
                        let _e565_ = (_e465_ + 3u);
                        if _e532_ {
                            phi_1871_ = 0u;
                        } else {
                            let _e566_ = (_e81_ - 1u);
                            phi_1871_ = select(_e566_, _e565_, (_e565_ < _e566_));
                        }
                        let _e570_ = phi_1871_;
                        if (_e570_ < _e81_) {
                        } else {
                            break;
                        }
                        ozaki_buckets.member[_e570_] = bitcast<i32>(_e528_.member_1_);
                    }
                }
            }
            break;
        }
    }
    return;
}

@compute @workgroup_size(256, 1, 1) 
fn ozaki_gram_rhs_i32(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    wg_id_1_ = wg_id;
    lid_1_ = lid;
    function_();
    return;
}

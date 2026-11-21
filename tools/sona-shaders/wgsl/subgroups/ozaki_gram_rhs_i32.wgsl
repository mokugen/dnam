struct type_6_ {
    member: array<f32>,
}

struct type_10_ {
    member: array<vec2<f32>>,
}

struct type_16_ {
    member: u32,
    member_1_: u32,
}

struct type_17_ {
    member: f32,
    member_1_: f32,
}

struct type_18_ {
    member: u32,
    member_1_: u32,
    member_2_: u32,
    member_3_: u32,
    member_4_: u32,
    member_5_: u32,
    member_6_: u32,
    member_7_: u32,
}

struct type_19_ {
    member: type_18_,
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
var<private> sg_lid_1_: u32;
@group(0) @binding(0) 
var<uniform> uniforms: type_19_;
@group(0) @binding(1) 
var<storage> features: type_6_;
@group(0) @binding(2) 
var<storage> targets: type_6_;
@group(0) @binding(3) 
var<storage> col_scales: type_10_;
@group(0) @binding(4) 
var<storage, read_write> ozaki_buckets: type_22_;
var<workgroup> reduce_i32_: array<i32, 1280>;

fn function_() {
    var local: array<i32, 5>;
    var local_1_: array<i32, 5>;
    var phi_216_: u32;
    var phi_219_: u32;
    var phi_237_: u32;
    var phi_217_: u32;
    var phi_220_: u32;
    var local_2_: u32;
    var local_3_: u32;
    var local_4_: u32;
    var phi_243_: type_28_;
    var phi_244_: type_28_;
    var phi_245_: bool;
    var phi_250_: type_28_;
    var phi_169_: u32;
    var phi_172_: u32;
    var phi_194_: u32;
    var phi_170_: u32;
    var phi_173_: u32;
    var local_5_: u32;
    var local_6_: u32;
    var local_7_: u32;
    var local_8_: u32;
    var phi_251_: type_28_;
    var phi_252_: type_28_;
    var phi_253_: bool;
    var phi_257_: type_28_;
    var phi_258_: type_28_;
    var phi_270_: u32;
    var phi_292_: u32;
    var phi_334_: bool;
    var phi_335_: bool;
    var phi_336_: f32;
    var phi_338_: bool;
    var phi_339_: f32;
    var phi_340_: bool;
    var phi_341_: bool;
    var phi_348_: f32;
    var phi_1256_: bool;
    var phi_385_: u32;
    var phi_392_: type_17_;
    var phi_393_: type_17_;
    var phi_394_: bool;
    var phi_395_: type_17_;
    var phi_396_: bool;
    var phi_401_: type_17_;
    var phi_403_: f32;
    var phi_435_: bool;
    var phi_436_: bool;
    var phi_437_: f32;
    var phi_439_: bool;
    var phi_440_: f32;
    var phi_441_: bool;
    var phi_442_: bool;
    var phi_449_: f32;
    var phi_1271_: bool;
    var phi_486_: u32;
    var phi_493_: type_17_;
    var phi_494_: type_17_;
    var phi_495_: bool;
    var phi_496_: type_17_;
    var phi_497_: bool;
    var phi_502_: type_17_;
    var phi_504_: f32;
    var phi_511_: bool;
    var phi_514_: type_16_;
    var phi_517_: array<i32, 6>;
    var phi_515_: type_16_;
    var phi_540_: type_16_;
    var phi_575_: u32;
    var phi_621_: u32;
    var phi_600_: u32;
    var phi_629_: f32;
    var phi_704_: i32;
    var phi_734_: i32;
    var phi_764_: i32;
    var phi_767_: array<i32, 3>;
    var phi_768_: array<i32, 3>;
    var phi_770_: array<i32, 3>;
    var phi_771_: bool;
    var phi_776_: array<i32, 3>;
    var phi_851_: i32;
    var phi_881_: i32;
    var phi_911_: i32;
    var phi_914_: array<i32, 3>;
    var phi_915_: array<i32, 3>;
    var phi_917_: array<i32, 3>;
    var phi_918_: bool;
    var phi_923_: array<i32, 3>;
    var phi_970_: array<i32, 6>;
    var phi_971_: array<i32, 6>;
    var phi_972_: array<i32, 6>;
    var phi_518_: array<i32, 6>;
    var phi_1276_: bool;
    var local_9_: array<i32, 6>;
    var local_10_: array<i32, 6>;
    var local_11_: array<i32, 6>;
    var local_12_: array<i32, 6>;
    var local_13_: array<i32, 6>;
    var phi_994_: type_16_;
    var phi_995_: type_16_;
    var phi_1018_: type_16_;
    var phi_1373_: bool;
    var phi_1384_: bool;
    var phi_1053_: u32;
    var phi_1068_: type_16_;
    var phi_1069_: type_16_;
    var phi_1092_: type_16_;
    var phi_1380_: bool;
    var phi_1387_: bool;
    var phi_1386_: bool;
    var phi_1054_: u32;
    var phi_1385_: bool;
    var phi_1156_: type_16_;
    var phi_1157_: type_16_;
    var phi_1180_: type_16_;
    var phi_1206_: u32;
    var phi_1388_: bool;
    var phi_1224_: u32;

    switch bitcast<i32>(0u) {
        default: {
            let _e68_ = wg_id_1_;
            let _e69_ = lid_1_;
            let _e70_ = sg_lid_1_;
            let _e72_ = arrayLength((&features.member));
            let _e74_ = arrayLength((&targets.member));
            let _e76_ = arrayLength((&col_scales.member));
            let _e78_ = arrayLength((&ozaki_buckets.member));
            let _e83_ = uniforms.member.member;
            let _e86_ = uniforms.member.member_1_;
            let _e89_ = uniforms.member.member_2_;
            let _e92_ = uniforms.member.member_3_;
            let _e94_ = select(_e92_, 1024u, (1024u < _e92_));
            let _e97_ = uniforms.member.member_4_;
            let _e99_ = select(1u, _e97_, (1u < _e97_));
            if (_e86_ == 0u) {
                phi_258_ = type_28_(0u, 0u, 0u, 0u, false, true);
            } else {
                let _e104_ = ((_e86_ * (_e86_ + 1u)) >> bitcast<u32>(1i));
                if (_e68_.x < _e104_) {
                    phi_169_ = (_e86_ - 1u);
                    phi_172_ = 0u;
                    loop {
                        let _e152_ = phi_169_;
                        let _e154_ = phi_172_;
                        local_5_ = _e154_;
                        local_6_ = _e154_;
                        local_7_ = _e154_;
                        local_8_ = _e154_;
                        let _e155_ = (_e154_ < _e152_);
                        if _e155_ {
                            let _e160_ = (_e154_ + (((_e152_ - _e154_) + 1u) >> bitcast<u32>(1i)));
                            let _e161_ = (_e86_ - _e160_);
                            let _e167_ = ((_e104_ - ((_e161_ * (_e161_ + 1u)) >> bitcast<u32>(1i))) <= _e68_.x);
                            if _e167_ {
                                phi_194_ = _e152_;
                            } else {
                                phi_194_ = (_e160_ - 1u);
                            }
                            let _e170_ = phi_194_;
                            phi_170_ = _e170_;
                            phi_173_ = select(_e154_, _e160_, _e167_);
                        } else {
                            phi_170_ = u32();
                            phi_173_ = u32();
                        }
                        let _e173_ = phi_170_;
                        let _e175_ = phi_173_;
                        phi_169_ = _e173_;
                        phi_172_ = _e175_;
                        if !(_e155_) {
                            break;
                        }
                    }
                    let _e178_ = local_5_;
                    let _e179_ = (_e86_ - _e178_);
                    let _e187_ = local_6_;
                    let _e188_ = (_e187_ + (_e68_.x - (_e104_ - ((_e179_ * (_e179_ + 1u)) >> bitcast<u32>(1i)))));
                    let _e190_ = local_7_;
                    let _e192_ = local_8_;
                    phi_251_ = type_28_(_e190_, _e188_, _e192_, _e188_, true, true);
                    phi_252_ = type_28_();
                    phi_253_ = true;
                } else {
                    let _e106_ = (_e68_.x - _e104_);
                    if (_e89_ == 0u) {
                        phi_244_ = type_28_();
                        phi_245_ = true;
                    } else {
                        let _e109_ = (_e106_ >= (_e86_ * _e89_));
                        if _e109_ {
                            phi_243_ = type_28_();
                        } else {
                            phi_216_ = (_e86_ - 1u);
                            phi_219_ = 0u;
                            loop {
                                let _e112_ = phi_216_;
                                let _e114_ = phi_219_;
                                local_2_ = _e114_;
                                local_3_ = _e114_;
                                local_4_ = _e114_;
                                let _e115_ = (_e114_ < _e112_);
                                if _e115_ {
                                    let _e120_ = (_e114_ + (((_e112_ - _e114_) + 1u) >> bitcast<u32>(1i)));
                                    let _e122_ = ((_e120_ * _e89_) <= _e106_);
                                    if _e122_ {
                                        phi_237_ = _e112_;
                                    } else {
                                        phi_237_ = (_e120_ - 1u);
                                    }
                                    let _e125_ = phi_237_;
                                    phi_217_ = _e125_;
                                    phi_220_ = select(_e114_, _e120_, _e122_);
                                } else {
                                    phi_217_ = u32();
                                    phi_220_ = u32();
                                }
                                let _e128_ = phi_217_;
                                let _e130_ = phi_220_;
                                phi_216_ = _e128_;
                                phi_219_ = _e130_;
                                if !(_e115_) {
                                    break;
                                }
                            }
                            let _e133_ = local_2_;
                            let _e135_ = (_e106_ - (_e133_ * _e89_));
                            let _e138_ = local_3_;
                            let _e140_ = local_4_;
                            phi_243_ = type_28_(_e138_, _e135_, _e140_, (_e86_ + _e135_), true, false);
                        }
                        let _e143_ = phi_243_;
                        phi_244_ = _e143_;
                        phi_245_ = _e109_;
                    }
                    let _e145_ = phi_244_;
                    let _e147_ = phi_245_;
                    if _e147_ {
                        phi_250_ = type_28_(0u, 0u, 0u, 0u, false, false);
                    } else {
                        phi_250_ = type_28_();
                    }
                    let _e149_ = phi_250_;
                    phi_251_ = _e149_;
                    phi_252_ = _e145_;
                    phi_253_ = _e147_;
                }
                let _e195_ = phi_251_;
                let _e197_ = phi_252_;
                let _e199_ = phi_253_;
                if _e199_ {
                    phi_257_ = _e195_;
                } else {
                    phi_257_ = _e197_;
                }
                let _e201_ = phi_257_;
                phi_258_ = _e201_;
            }
            let _e203_ = phi_258_;
            let _e205_ = (_e76_ == 0u);
            if _e205_ {
                phi_270_ = 0u;
            } else {
                let _e206_ = (_e76_ - 1u);
                phi_270_ = select(_e206_, _e203_.member_2_, (_e203_.member_2_ < _e206_));
            }
            let _e210_ = phi_270_;
            if (_e210_ < _e76_) {
            } else {
                break;
            }
            let _e215_ = col_scales.member[_e210_].x;
            let _e219_ = col_scales.member[_e210_].y;
            if _e205_ {
                phi_292_ = 0u;
            } else {
                let _e221_ = (_e76_ - 1u);
                phi_292_ = select(_e221_, _e203_.member_3_, (_e203_.member_3_ < _e221_));
            }
            let _e225_ = phi_292_;
            if (_e225_ < _e76_) {
            } else {
                break;
            }
            let _e230_ = col_scales.member[_e225_].x;
            let _e234_ = col_scales.member[_e225_].y;
            let _e235_ = bitcast<u32>(_e219_);
            let _e238_ = ((_e235_ >> bitcast<u32>(23u)) & 255u);
            if ((_e235_ & 2155872255u) == 0u) {
                let _e241_ = (_e238_ > 0u);
                if _e241_ {
                    let _e243_ = bitcast<u32>(_e215_);
                    let _e245_ = bitcast<f32>((_e243_ & 2147483647u));
                    if (_e238_ < 255u) {
                        if ((_e243_ & 2139095040u) != 2139095040u) {
                            phi_334_ = select(true, false, ((_e245_ * _e219_) <= 1f));
                        } else {
                            phi_334_ = true;
                        }
                        let _e252_ = phi_334_;
                        phi_335_ = _e252_;
                    } else {
                        phi_335_ = true;
                    }
                    let _e254_ = phi_335_;
                    phi_336_ = _e245_;
                    phi_338_ = _e254_;
                } else {
                    phi_336_ = f32();
                    phi_338_ = false;
                }
                let _e256_ = phi_336_;
                let _e258_ = phi_338_;
                phi_339_ = _e256_;
                phi_340_ = select(true, false, _e241_);
                phi_341_ = _e258_;
            } else {
                phi_339_ = f32();
                phi_340_ = true;
                phi_341_ = false;
            }
            let _e261_ = phi_339_;
            let _e263_ = phi_340_;
            let _e265_ = phi_341_;
            if _e263_ {
                phi_348_ = bitcast<f32>((bitcast<u32>(_e215_) & 2147483647u));
            } else {
                phi_348_ = _e261_;
            }
            let _e270_ = phi_348_;
            if select(_e265_, true, _e263_) {
                let _e272_ = bitcast<u32>(_e270_);
                if ((_e272_ & 2147483648u) == 0u) {
                    if (_e272_ == 0u) {
                        phi_393_ = type_17_();
                        phi_394_ = true;
                    } else {
                        let _e276_ = (_e272_ >= 2139095040u);
                        if _e276_ {
                            phi_392_ = type_17_();
                        } else {
                            if (_e270_ != _e270_) {
                                phi_1256_ = true;
                            } else {
                                phi_1256_ = (85070590000000000000000000000000000000f <= _e270_);
                            }
                            let _e280_ = phi_1256_;
                            let _e286_ = (((bitcast<u32>(select(_e270_, 85070590000000000000000000000000000000f, _e280_)) + 8388607u) & 2139095040u) >> bitcast<u32>(23u));
                            if (_e286_ < 1u) {
                                phi_385_ = 1u;
                            } else {
                                phi_385_ = select(_e286_, 253u, (_e286_ > 253u));
                            }
                            let _e291_ = phi_385_;
                            phi_392_ = type_17_(bitcast<f32>((_e291_ << bitcast<u32>(23u))), bitcast<f32>(((254u - _e291_) << bitcast<u32>(23u))));
                        }
                        let _e301_ = phi_392_;
                        phi_393_ = _e301_;
                        phi_394_ = _e276_;
                    }
                    let _e303_ = phi_393_;
                    let _e305_ = phi_394_;
                    phi_395_ = _e303_;
                    phi_396_ = _e305_;
                } else {
                    phi_395_ = type_17_();
                    phi_396_ = true;
                }
                let _e307_ = phi_395_;
                let _e309_ = phi_396_;
                if _e309_ {
                    phi_401_ = type_17_(1f, 1f);
                } else {
                    phi_401_ = _e307_;
                }
                let _e311_ = phi_401_;
                phi_403_ = _e311_.member_1_;
            } else {
                phi_403_ = _e219_;
            }
            let _e314_ = phi_403_;
            let _e315_ = bitcast<u32>(_e234_);
            let _e318_ = ((_e315_ >> bitcast<u32>(23u)) & 255u);
            if ((_e315_ & 2155872255u) == 0u) {
                let _e321_ = (_e318_ > 0u);
                if _e321_ {
                    let _e323_ = bitcast<u32>(_e230_);
                    let _e325_ = bitcast<f32>((_e323_ & 2147483647u));
                    if (_e318_ < 255u) {
                        if ((_e323_ & 2139095040u) != 2139095040u) {
                            phi_435_ = select(true, false, ((_e325_ * _e234_) <= 1f));
                        } else {
                            phi_435_ = true;
                        }
                        let _e332_ = phi_435_;
                        phi_436_ = _e332_;
                    } else {
                        phi_436_ = true;
                    }
                    let _e334_ = phi_436_;
                    phi_437_ = _e325_;
                    phi_439_ = _e334_;
                } else {
                    phi_437_ = f32();
                    phi_439_ = false;
                }
                let _e336_ = phi_437_;
                let _e338_ = phi_439_;
                phi_440_ = _e336_;
                phi_441_ = select(true, false, _e321_);
                phi_442_ = _e338_;
            } else {
                phi_440_ = f32();
                phi_441_ = true;
                phi_442_ = false;
            }
            let _e341_ = phi_440_;
            let _e343_ = phi_441_;
            let _e345_ = phi_442_;
            if _e343_ {
                phi_449_ = bitcast<f32>((bitcast<u32>(_e230_) & 2147483647u));
            } else {
                phi_449_ = _e341_;
            }
            let _e350_ = phi_449_;
            if select(_e345_, true, _e343_) {
                let _e352_ = bitcast<u32>(_e350_);
                if ((_e352_ & 2147483648u) == 0u) {
                    if (_e352_ == 0u) {
                        phi_494_ = type_17_();
                        phi_495_ = true;
                    } else {
                        let _e356_ = (_e352_ >= 2139095040u);
                        if _e356_ {
                            phi_493_ = type_17_();
                        } else {
                            if (_e350_ != _e350_) {
                                phi_1271_ = true;
                            } else {
                                phi_1271_ = (85070590000000000000000000000000000000f <= _e350_);
                            }
                            let _e360_ = phi_1271_;
                            let _e366_ = (((bitcast<u32>(select(_e350_, 85070590000000000000000000000000000000f, _e360_)) + 8388607u) & 2139095040u) >> bitcast<u32>(23u));
                            if (_e366_ < 1u) {
                                phi_486_ = 1u;
                            } else {
                                phi_486_ = select(_e366_, 253u, (_e366_ > 253u));
                            }
                            let _e371_ = phi_486_;
                            phi_493_ = type_17_(bitcast<f32>((_e371_ << bitcast<u32>(23u))), bitcast<f32>(((254u - _e371_) << bitcast<u32>(23u))));
                        }
                        let _e381_ = phi_493_;
                        phi_494_ = _e381_;
                        phi_495_ = _e356_;
                    }
                    let _e383_ = phi_494_;
                    let _e385_ = phi_495_;
                    phi_496_ = _e383_;
                    phi_497_ = _e385_;
                } else {
                    phi_496_ = type_17_();
                    phi_497_ = true;
                }
                let _e387_ = phi_496_;
                let _e389_ = phi_497_;
                if _e389_ {
                    phi_502_ = type_17_(1f, 1f);
                } else {
                    phi_502_ = _e387_;
                }
                let _e391_ = phi_502_;
                phi_504_ = _e391_.member_1_;
            } else {
                phi_504_ = _e234_;
            }
            let _e394_ = phi_504_;
            if _e203_.member_4_ {
                phi_511_ = (_e68_.y < _e99_);
            } else {
                phi_511_ = false;
            }
            let _e399_ = phi_511_;
            phi_514_ = type_16_(0u, 4u);
            phi_517_ = array<i32, 6>(0i, 0i, 0i, 0i, 0i, 0i);
            loop {
                let _e401_ = phi_514_;
                let _e403_ = phi_517_;
                local_9_ = _e403_;
                local_10_ = _e403_;
                local_11_ = _e403_;
                local_12_ = _e403_;
                local_13_ = _e403_;
                if (_e401_.member < _e401_.member_1_) {
                    phi_515_ = type_16_((_e401_.member + 1u), _e401_.member_1_);
                    phi_540_ = type_16_(1u, _e401_.member);
                } else {
                    phi_515_ = _e401_;
                    phi_540_ = type_16_(0u, u32());
                }
                let _e416_ = phi_515_;
                let _e418_ = phi_540_;
                let _e422_ = (bitcast<i32>(_e418_.member) != 0i);
                if _e422_ {
                    let _e424_ = ((_e418_.member_1_ * 256u) + _e69_);
                    let _e425_ = ((_e68_.y * _e94_) + _e424_);
                    if _e399_ {
                        if (_e424_ < _e94_) {
                            if (_e425_ < _e83_) {
                                let _e430_ = ((_e203_.member * _e83_) + _e425_);
                                let _e431_ = (_e72_ == 0u);
                                if _e431_ {
                                    phi_575_ = 0u;
                                } else {
                                    let _e432_ = (_e72_ - 1u);
                                    phi_575_ = select(_e432_, _e430_, (_e430_ < _e432_));
                                }
                                let _e436_ = phi_575_;
                                if (_e436_ < _e72_) {
                                } else {
                                    phi_1276_ = true;
                                    break;
                                }
                                let _e440_ = features.member[_e436_];
                                if _e203_.member_5_ {
                                    let _e457_ = ((_e203_.member_1_ * _e83_) + _e425_);
                                    if _e431_ {
                                        phi_600_ = 0u;
                                    } else {
                                        let _e458_ = (_e72_ - 1u);
                                        phi_600_ = select(_e458_, _e457_, (_e457_ < _e458_));
                                    }
                                    let _e462_ = phi_600_;
                                    if (_e462_ < _e72_) {
                                    } else {
                                        phi_1276_ = true;
                                        break;
                                    }
                                    let _e466_ = features.member[_e462_];
                                    phi_629_ = _e466_;
                                } else {
                                    let _e444_ = ((_e203_.member_1_ * _e83_) + _e425_);
                                    if (_e74_ == 0u) {
                                        phi_621_ = 0u;
                                    } else {
                                        let _e446_ = (_e74_ - 1u);
                                        phi_621_ = select(_e446_, _e444_, (_e444_ < _e446_));
                                    }
                                    let _e450_ = phi_621_;
                                    if (_e450_ < _e74_) {
                                    } else {
                                        phi_1276_ = true;
                                        break;
                                    }
                                    let _e454_ = targets.member[_e450_];
                                    phi_629_ = _e454_;
                                }
                                let _e468_ = phi_629_;
                                if ((bitcast<u32>(_e440_) & 2139095040u) != 2139095040u) {
                                    let _e474_ = ((bitcast<u32>(_e314_) & 2139095040u) != 2139095040u);
                                    if _e474_ {
                                        let _e476_ = select(_e440_, -85070590000000000000000000000000000000f, (_e440_ < -85070590000000000000000000000000000000f));
                                        let _e479_ = (select(_e476_, 85070590000000000000000000000000000000f, (_e476_ > 85070590000000000000000000000000000000f)) * _e314_);
                                        if ((bitcast<u32>(_e479_) & 2139095040u) != 2139095040u) {
                                            let _e484_ = select(_e479_, -1f, (_e479_ < -1f));
                                            let _e486_ = select(_e484_, 1f, (_e484_ > 1f));
                                            let _e487_ = (_e486_ * 256f);
                                            if ((bitcast<u32>(_e487_) & 2139095040u) != 2139095040u) {
                                                let _e492_ = select(_e487_, -4194304f, (_e487_ < -4194304f));
                                                phi_704_ = (bitcast<i32>(bitcast<u32>((select(_e492_, 4194304f, (_e492_ > 4194304f)) + 12582912f))) - 1262485504i);
                                            } else {
                                                phi_704_ = 0i;
                                            }
                                            let _e500_ = phi_704_;
                                            let _e503_ = (_e486_ - (f32(_e500_) * 0.00390625f));
                                            let _e504_ = (_e503_ * 65536f);
                                            if ((bitcast<u32>(_e504_) & 2139095040u) != 2139095040u) {
                                                let _e509_ = select(_e504_, -4194304f, (_e504_ < -4194304f));
                                                phi_734_ = (bitcast<i32>(bitcast<u32>((select(_e509_, 4194304f, (_e509_ > 4194304f)) + 12582912f))) - 1262485504i);
                                            } else {
                                                phi_734_ = 0i;
                                            }
                                            let _e517_ = phi_734_;
                                            let _e521_ = ((_e503_ - (f32(_e517_) * 0.000015258789f)) * 16777216f);
                                            if ((bitcast<u32>(_e521_) & 2139095040u) != 2139095040u) {
                                                let _e526_ = select(_e521_, -4194304f, (_e521_ < -4194304f));
                                                phi_764_ = (bitcast<i32>(bitcast<u32>((select(_e526_, 4194304f, (_e526_ > 4194304f)) + 12582912f))) - 1262485504i);
                                            } else {
                                                phi_764_ = 0i;
                                            }
                                            let _e534_ = phi_764_;
                                            phi_767_ = array<i32, 3>(_e500_, _e517_, _e534_);
                                        } else {
                                            phi_767_ = array<i32, 3>(0i, 0i, 0i);
                                        }
                                        let _e537_ = phi_767_;
                                        phi_768_ = _e537_;
                                    } else {
                                        phi_768_ = array<i32, 3>();
                                    }
                                    let _e539_ = phi_768_;
                                    phi_770_ = _e539_;
                                    phi_771_ = select(true, false, _e474_);
                                } else {
                                    phi_770_ = array<i32, 3>();
                                    phi_771_ = true;
                                }
                                let _e542_ = phi_770_;
                                let _e544_ = phi_771_;
                                if _e544_ {
                                    phi_776_ = array<i32, 3>(0i, 0i, 0i);
                                } else {
                                    phi_776_ = _e542_;
                                }
                                let _e546_ = phi_776_;
                                if ((bitcast<u32>(_e468_) & 2139095040u) != 2139095040u) {
                                    let _e552_ = ((bitcast<u32>(_e394_) & 2139095040u) != 2139095040u);
                                    if _e552_ {
                                        let _e554_ = select(_e468_, -85070590000000000000000000000000000000f, (_e468_ < -85070590000000000000000000000000000000f));
                                        let _e557_ = (select(_e554_, 85070590000000000000000000000000000000f, (_e554_ > 85070590000000000000000000000000000000f)) * _e394_);
                                        if ((bitcast<u32>(_e557_) & 2139095040u) != 2139095040u) {
                                            let _e562_ = select(_e557_, -1f, (_e557_ < -1f));
                                            let _e564_ = select(_e562_, 1f, (_e562_ > 1f));
                                            let _e565_ = (_e564_ * 256f);
                                            if ((bitcast<u32>(_e565_) & 2139095040u) != 2139095040u) {
                                                let _e570_ = select(_e565_, -4194304f, (_e565_ < -4194304f));
                                                phi_851_ = (bitcast<i32>(bitcast<u32>((select(_e570_, 4194304f, (_e570_ > 4194304f)) + 12582912f))) - 1262485504i);
                                            } else {
                                                phi_851_ = 0i;
                                            }
                                            let _e578_ = phi_851_;
                                            let _e581_ = (_e564_ - (f32(_e578_) * 0.00390625f));
                                            let _e582_ = (_e581_ * 65536f);
                                            if ((bitcast<u32>(_e582_) & 2139095040u) != 2139095040u) {
                                                let _e587_ = select(_e582_, -4194304f, (_e582_ < -4194304f));
                                                phi_881_ = (bitcast<i32>(bitcast<u32>((select(_e587_, 4194304f, (_e587_ > 4194304f)) + 12582912f))) - 1262485504i);
                                            } else {
                                                phi_881_ = 0i;
                                            }
                                            let _e595_ = phi_881_;
                                            let _e599_ = ((_e581_ - (f32(_e595_) * 0.000015258789f)) * 16777216f);
                                            if ((bitcast<u32>(_e599_) & 2139095040u) != 2139095040u) {
                                                let _e604_ = select(_e599_, -4194304f, (_e599_ < -4194304f));
                                                phi_911_ = (bitcast<i32>(bitcast<u32>((select(_e604_, 4194304f, (_e604_ > 4194304f)) + 12582912f))) - 1262485504i);
                                            } else {
                                                phi_911_ = 0i;
                                            }
                                            let _e612_ = phi_911_;
                                            phi_914_ = array<i32, 3>(_e578_, _e595_, _e612_);
                                        } else {
                                            phi_914_ = array<i32, 3>(0i, 0i, 0i);
                                        }
                                        let _e615_ = phi_914_;
                                        phi_915_ = _e615_;
                                    } else {
                                        phi_915_ = array<i32, 3>();
                                    }
                                    let _e617_ = phi_915_;
                                    phi_917_ = _e617_;
                                    phi_918_ = select(true, false, _e552_);
                                } else {
                                    phi_917_ = array<i32, 3>();
                                    phi_918_ = true;
                                }
                                let _e620_ = phi_917_;
                                let _e622_ = phi_918_;
                                if _e622_ {
                                    phi_923_ = array<i32, 3>(0i, 0i, 0i);
                                } else {
                                    phi_923_ = _e620_;
                                }
                                let _e624_ = phi_923_;
                                let _e636_ = array<i32, 6>((_e403_[0] + (_e546_[0] * _e624_[0])), _e403_[1], _e403_[2], _e403_[3], _e403_[4], _e403_[5]);
                                let _e650_ = array<i32, 6>(_e636_[0], (_e403_[1] + ((_e546_[0] * _e624_[1]) + (_e546_[1] * _e624_[0]))), _e636_[2], _e636_[3], _e636_[4], _e636_[5]);
                                let _e666_ = array<i32, 6>(_e650_[0], _e650_[1], (_e403_[2] + (((_e546_[0] * _e624_[2]) + (_e546_[1] * _e624_[1])) + (_e546_[2] * _e624_[0]))), _e650_[3], _e650_[4], _e650_[5]);
                                let _e678_ = array<i32, 6>(_e666_[0], _e666_[1], _e666_[2], (_e403_[3] + ((_e546_[1] * _e624_[2]) + (_e546_[2] * _e624_[1]))), _e666_[4], _e666_[5]);
                                phi_970_ = array<i32, 6>(_e678_[0], _e678_[1], _e678_[2], _e678_[3], (_e403_[4] + (_e546_[2] * _e624_[2])), _e678_[5]);
                            } else {
                                phi_970_ = _e403_;
                            }
                            let _e690_ = phi_970_;
                            phi_971_ = _e690_;
                        } else {
                            phi_971_ = _e403_;
                        }
                        let _e692_ = phi_971_;
                        phi_972_ = _e692_;
                    } else {
                        phi_972_ = _e403_;
                    }
                    let _e694_ = phi_972_;
                    phi_518_ = _e694_;
                } else {
                    phi_518_ = array<i32, 6>();
                }
                let _e696_ = phi_518_;
                phi_514_ = _e416_;
                phi_517_ = _e696_;
                phi_1276_ = false;
                if !(_e422_) {
                    break;
                }
            }
            let _e699_ = phi_1276_;
            if _e699_ {
                break;
            }
            let _e701_ = local_9_;
            let _e704_ = local_10_;
            let _e707_ = local_11_;
            let _e710_ = local_12_;
            let _e713_ = local_13_;
            let _e997 = subgroupAdd(_e701_[0]);
            let _e999 = subgroupAdd(_e704_[1]);
            let _e1001 = subgroupAdd(_e707_[2]);
            let _e1003 = subgroupAdd(_e710_[3]);
            let _e1005 = subgroupAdd(_e713_[4]);
            if (_e70_ == 0u) {
                local[0] = _e997;
                local[1] = _e999;
                local[2] = _e1001;
                local[3] = _e1003;
                local[4] = _e1005;
            } else {
                local = array<i32, 5>(0i, 0i, 0i, 0i, 0i);
            }
            phi_994_ = type_16_(0u, 5u);
            loop {
                let _e727_ = phi_994_;
                if (_e727_.member < _e727_.member_1_) {
                    phi_995_ = type_16_((_e727_.member + 1u), _e727_.member_1_);
                    phi_1018_ = type_16_(1u, _e727_.member);
                } else {
                    phi_995_ = _e727_;
                    phi_1018_ = type_16_(0u, u32());
                }
                let _e740_ = phi_995_;
                let _e742_ = phi_1018_;
                let _e746_ = (bitcast<i32>(_e742_.member) != 0i);
                if _e746_ {
                    if (_e742_.member_1_ < 5u) {
                    } else {
                        phi_1373_ = true;
                        break;
                    }
                    let _e749_ = local[_e742_.member_1_];
                    let _e751_ = ((_e742_.member_1_ * 256u) + _e69_);
                    let _e753_ = select(1279u, _e751_, (_e751_ < 1279u));
                    if (_e753_ < 1280u) {
                    } else {
                        phi_1373_ = true;
                        break;
                    }
                    reduce_i32_[_e753_] = _e749_;
                }
                phi_994_ = _e740_;
                phi_1373_ = _e699_;
                if !(_e746_) {
                    break;
                }
            }
            let _e758_ = phi_1373_;
            if _e758_ {
                break;
            }
            workgroupBarrier();
            phi_1384_ = _e758_;
            phi_1053_ = 128u;
            loop {
                let _e760_ = phi_1384_;
                let _e762_ = phi_1053_;
                let _e763_ = (_e762_ > 0u);
                if _e763_ {
                    if (_e69_ < _e762_) {
                        phi_1068_ = type_16_(0u, 5u);
                        loop {
                            let _e766_ = phi_1068_;
                            if (_e766_.member < _e766_.member_1_) {
                                phi_1069_ = type_16_((_e766_.member + 1u), _e766_.member_1_);
                                phi_1092_ = type_16_(1u, _e766_.member);
                            } else {
                                phi_1069_ = _e766_;
                                phi_1092_ = type_16_(0u, u32());
                            }
                            let _e779_ = phi_1069_;
                            let _e781_ = phi_1092_;
                            let _e785_ = (bitcast<i32>(_e781_.member) != 0i);
                            if _e785_ {
                                let _e787_ = ((_e781_.member_1_ * 256u) + _e69_);
                                let _e788_ = (_e787_ < 1280u);
                                if _e788_ {
                                } else {
                                    phi_1380_ = true;
                                    break;
                                }
                                let _e790_ = reduce_i32_[_e787_];
                                let _e791_ = (_e787_ + _e762_);
                                if (_e791_ < 1280u) {
                                } else {
                                    phi_1380_ = true;
                                    break;
                                }
                                let _e794_ = reduce_i32_[_e791_];
                                if _e788_ {
                                } else {
                                    phi_1380_ = true;
                                    break;
                                }
                                reduce_i32_[_e787_] = (_e790_ + _e794_);
                            }
                            phi_1068_ = _e779_;
                            phi_1380_ = _e760_;
                            if !(_e785_) {
                                break;
                            }
                        }
                        let _e798_ = phi_1380_;
                        phi_1385_ = _e798_;
                        if _e798_ {
                            break;
                        }
                        phi_1387_ = _e798_;
                    } else {
                        phi_1387_ = _e760_;
                    }
                    let _e800_ = phi_1387_;
                    workgroupBarrier();
                    phi_1386_ = _e800_;
                    phi_1054_ = (_e762_ >> bitcast<u32>(1i));
                } else {
                    phi_1386_ = _e760_;
                    phi_1054_ = u32();
                }
                let _e804_ = phi_1386_;
                let _e806_ = phi_1054_;
                phi_1384_ = _e804_;
                phi_1053_ = _e806_;
                phi_1385_ = _e804_;
                if !(_e763_) {
                    break;
                }
            }
            let _e809_ = phi_1385_;
            if _e809_ {
                break;
            }
            let _e811_ = reduce_i32_[0];
            let _e813_ = reduce_i32_[256];
            let _e815_ = reduce_i32_[512];
            let _e817_ = reduce_i32_[768];
            let _e819_ = reduce_i32_[1024];
            workgroupBarrier();
            local_1_ = array<i32, 5>(_e811_, _e813_, _e815_, _e817_, _e819_);
            if (_e69_ == 0u) {
                if _e399_ {
                    let _e824_ = (((_e68_.x * _e99_) + _e68_.y) * 6u);
                    phi_1156_ = type_16_(0u, 5u);
                    loop {
                        let _e826_ = phi_1156_;
                        if (_e826_.member < _e826_.member_1_) {
                            phi_1157_ = type_16_((_e826_.member + 1u), _e826_.member_1_);
                            phi_1180_ = type_16_(1u, _e826_.member);
                        } else {
                            phi_1157_ = _e826_;
                            phi_1180_ = type_16_(0u, u32());
                        }
                        let _e839_ = phi_1157_;
                        let _e841_ = phi_1180_;
                        let _e845_ = (bitcast<i32>(_e841_.member) != 0i);
                        if _e845_ {
                            if (_e841_.member_1_ < 5u) {
                            } else {
                                phi_1388_ = true;
                                break;
                            }
                            let _e848_ = local_1_[_e841_.member_1_];
                            let _e849_ = (_e824_ + _e841_.member_1_);
                            if (_e78_ == 0u) {
                                phi_1206_ = 0u;
                            } else {
                                let _e851_ = (_e78_ - 1u);
                                phi_1206_ = select(_e851_, _e849_, (_e849_ < _e851_));
                            }
                            let _e855_ = phi_1206_;
                            if (_e855_ < _e78_) {
                            } else {
                                phi_1388_ = true;
                                break;
                            }
                            ozaki_buckets.member[_e855_] = _e848_;
                        }
                        phi_1156_ = _e839_;
                        phi_1388_ = _e809_;
                        if !(_e845_) {
                            break;
                        }
                    }
                    let _e861_ = phi_1388_;
                    if _e861_ {
                        break;
                    }
                    let _e862_ = (_e824_ + 5u);
                    if (_e78_ == 0u) {
                        phi_1224_ = 0u;
                    } else {
                        let _e864_ = (_e78_ - 1u);
                        phi_1224_ = select(_e864_, _e862_, (_e862_ < _e864_));
                    }
                    let _e868_ = phi_1224_;
                    if (_e868_ < _e78_) {
                    } else {
                        break;
                    }
                    ozaki_buckets.member[_e868_] = 0i;
                }
            }
            break;
        }
    }
    return;
}

@compute @workgroup_size(256, 1, 1) 
fn ozaki_gram_rhs_i32(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_index) lid: u32, @builtin(subgroup_invocation_id) sg_lid: u32) {
    wg_id_1_ = wg_id;
    lid_1_ = lid;
    sg_lid_1_ = sg_lid;
    function_();
    return;
}
